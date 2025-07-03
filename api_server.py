
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from fastapi import HTTPException

from pydantic import BaseModel
from typing import List, Optional

import logging
from pathlib import Path
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import argparse
import json
import asyncio
import time
import numpy as np
import soundfile as sf
import uuid
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks

from indextts.infer_vllm import IndexTTS

tts = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("index-tts-vllm-api") # 建议添加日志


# 目录配置
OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "outputs")
REFERENCE_DIR = os.environ.get("TTS_REFERENCE_DIR", "references")
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "checkpoints")
TEMP_DIR = os.environ.get("TTS_TEMP_DIR", "temp_uploads")
BPE_PATH = os.environ.get("TTS_BPE_PATH", f"{MODEL_DIR}/bpe_cn_en.model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 定义请求模型
class TTSRequest(BaseModel):
    text: str
    reference_id: str
    temperature: float = 1.0
    top_p: float = 0.8
    speed: float = 1.0
    volume: float = 1.0
    pitch: float = 0.0
    fusion_method: str = "average"
    weights: Optional[List[float]] = None
    no_chunk: bool = False
    stream: bool = False

# 定义响应模型
class TTSResponse(BaseModel):
    id: str
    audio_url: str
    duration: float
    text: str
    sampling_rate: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    cfg_path = os.path.join(args.model_dir, "config.yaml")
    tts = IndexTTS(model_dir=args.model_dir, cfg_path=cfg_path, gpu_memory_utilization=args.gpu_memory_utilization)

    logger.info(f"正在从 '{REFERENCE_DIR}' 目录扫描并注册音色...")
    if not os.path.exists(REFERENCE_DIR):
        os.makedirs(REFERENCE_DIR)
        logger.warning(f"参考音频目录 '{REFERENCE_DIR}' 不存在，已自动创建。")

    speaker_count = 0
    for speaker_dir in Path(REFERENCE_DIR).iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            audio_files = [str(p) for p in speaker_dir.glob("*")
                           if p.suffix.lower() in ['.wav', '.mp3', '.flac']]

            if audio_files:
                # 将会使用音频文件列表中的第一个
                tts.registry_speaker(speaker_id, audio_files)
                logger.info(f"成功注册音色: '{speaker_id}' (使用单个音频文件)。")
                speaker_count += 1

    if speaker_count == 0:
        logger.warning(f"警告: '{REFERENCE_DIR}' 中没有找到任何可用的音色。")

    logger.info("Application startup complete.")
    yield
    # Clean up the ML models and release the resources
    # ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        audio_paths = data["audio_paths"]

        global tts
        sr, wav = await tts.infer(audio_paths, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )

# 临时存储任务结果，用于后台清理
results = {}

@app.post("/v1/tts", response_model=TTSResponse, tags=["Compatibility Endpoints"])
async def compatible_generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    [兼容旧版] 异步生成TTS，返回包含音频URL的JSON响应。
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received compatible request {task_id} for speaker '{request.reference_id}'")

    if tts is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")

    try:
        # --- 适配层：转换输入参数 ---
        character = request.reference_id.split(',')[0].strip()

        if character not in tts.speaker_dict:
            raise HTTPException(status_code=404, detail=f"Speaker '{character}' not found.")

        # --- 调用核心生成逻辑
        sr, wav = await tts.infer_with_ref_audio_embed(character, request.text)
        # --- 核心生成逻辑调用结束 ---

        # --- 适配层：转换输出结果 ---
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
        sf.write(output_path, wav, sr)

        duration = len(wav) / sr

        response_data = TTSResponse(
            id=task_id,
            audio_url=f"/v1/audio/{task_id}",
            duration=round(duration, 2),
            text=request.text,
            sampling_rate=sr
        )

        results[task_id] = response_data.model_dump()
        background_tasks.add_task(lambda: results.pop(task_id, None))

        return response_data

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/tts_audio", tags=["Compatibility Endpoints"])
async def compatible_generate_and_return_tts_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """[兼容旧版] 生成TTS并直接返回音频文件"""
    response_data = await compatible_generate_tts(request, background_tasks)
    task_id = response_data.id
    file_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generated audio file not found.")
    background_tasks.add_task(os.remove, file_path)
    return FileResponse(file_path, media_type="audio/wav", filename=f"{task_id}.wav")

@app.get("/v1/audio/{audio_id}", tags=["Compatibility Endpoints"])
async def compatible_get_audio(audio_id: str):
    """[兼容旧版] 获取生成的音频文件"""
    file_path = os.path.join(OUTPUT_DIR, f"{audio_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found or has been cleaned up.")
    return FileResponse(file_path, media_type="audio/wav", filename=f"{audio_id}.wav")

@app.get("/v1/references", tags=["Compatibility Endpoints"])
async def compatible_list_references():
    """[兼容旧版] 列出所有可用的参考音频ID (说话人)"""
    if tts is None or not hasattr(tts, 'speaker_dict'):
        return {"references": []}
    references = [{"id": spk_id, "name": spk_id} for spk_id in tts.speaker_dict.keys()]
    return {"references": references}

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "healthy", "version": "index-tts-vllm 1.5"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 保持新仓库的命令行参数风格，同时可以被环境变量覆盖
    default_model_dir = os.environ.get("TTS_MODEL_DIR", "/path/to/IndexTeam/Index-TTS")
    default_port = int(os.environ.get("SERVICE_PORT", 11996))

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--model_dir", type=str, default=default_model_dir)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    args = parser.parse_args()

    # 简单的启动前检查
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        logger.error("Please specify a valid path using --model_dir or the TTS_MODEL_DIR environment variable.")
    else:
        logger.info(f"Starting compatible IndexTTS API on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)