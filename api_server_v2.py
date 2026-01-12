import os
import asyncio
import io
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union
from pathlib import Path
import uuid
from pydantic import BaseModel, Field
import logging
import numpy as np
from starlette.background import BackgroundTask

from indextts.infer_vllm_v2 import IndexTTS2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("index-tts-vllm-v2-api")

# 目录配置
OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "outputs")
REFERENCE_DIR = os.environ.get("TTS_REFERENCE_DIR", "references")
MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "checkpoints")
TEMP_DIR = os.environ.get("TTS_TEMP_DIR", "temp_uploads")

# 确保目录存在
for dir_path in [OUTPUT_DIR, REFERENCE_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

tts = None

# 添加1.0版本兼容的请求模型
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

class TTSResponse(BaseModel):
    id: str
    audio_url: str
    duration: float
    text: str
    sampling_rate: int

class AudioTTSRequest(BaseModel):
    """使用参考音频文件的TTS请求"""
    text: str
    reference_audio_path: str  # 本地参考音频文件路径
    temperature: float = 1.0
    top_p: float = 0.8

    # IndexTTS2.0 情感控制参数
    emo_alpha: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    emo_vector: Optional[List[float]] = None
    use_emo_text: Optional[bool] = False
    emo_text: Optional[str] = None
    use_random: Optional[bool] = False
    interval_silence: Optional[int] = 200
    max_text_tokens_per_segment: Optional[int] = 120

    # 生成参数
    top_k: Optional[int] = 30
    repetition_penalty: Optional[float] = 10.0
    max_mel_tokens: Optional[int] = 1500

# 临时存储任务结果
results = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    # 从 app.state 中获取参数，或者使用环境变量作为后备
    model_dir = getattr(app.state, 'model_dir', MODEL_DIR)
    is_fp16 = getattr(app.state, 'is_fp16', False)
    gpu_memory_utilization = getattr(app.state, 'gpu_memory_utilization', 0.25)

    tts = IndexTTS2(model_dir=model_dir, is_fp16=is_fp16, gpu_memory_utilization=gpu_memory_utilization)

    # 自动注册音色（从1.0版本移植）
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
                tts.registry_speaker(speaker_id, audio_files)
                logger.info(f"成功注册音色: '{speaker_id}' (使用单个音频文件)。")
                speaker_count += 1

    if speaker_count == 0:
        logger.warning(f"警告: '{REFERENCE_DIR}' 中没有找到任何可用的音色。")

    logger.info(f"Application startup complete. Registered {speaker_count} speakers.")
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 保持原有的API
@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        print(f"Emo control mode:{emo_control_method},vec:{vec}")
        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                                  output_path=None,
                                  emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                                  emo_vector=vec,
                                  use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                                  max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))

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

# 添加1.0版本兼容的API接口

@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    """兼容1.0版本的 /tts 接口"""
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)

        # 确保 wav 是 numpy 数组格式
        if isinstance(wav, tuple) and len(wav) == 2:
            sr, wav = wav

        # 应用1.0版本的音频后处理
        if isinstance(wav, np.ndarray):
            wav_data = wav.astype(np.float32)
            if wav_data.ndim == 1:
                wav_data = wav_data.reshape(-1, 1)
        else:
            # 如果是 torch tensor
            wav_data = wav.cpu().numpy().astype(np.float32)
            if wav_data.ndim == 1:
                wav_data = wav_data.reshape(-1, 1)

        wav_data = tts.trim_and_pad_silence(wav_data)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav_data.flatten(), sr, format='WAV')
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

@app.post("/v1/tts", response_model=TTSResponse, tags=["Compatibility Endpoints"])
async def compatible_generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """[兼容1.0版本] 异步生成TTS，返回包含音频URL的JSON响应"""
    task_id = str(uuid.uuid4())
    logger.info(f"Received compatible request {task_id} for speaker '{request.reference_id}'")

    if tts is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")

    try:
        character = request.reference_id.split(',')[0].strip()

        if character not in tts.speaker_dict:
            raise HTTPException(status_code=404, detail=f"Speaker '{character}' not found.")

        # 调用兼容的推理方法
        sr, wav = await tts.infer_with_ref_audio_embed(character, request.text)

        # 确保 wav 是 numpy 数组格式
        if isinstance(wav, tuple) and len(wav) == 2:
            sr, wav = wav

        # 应用1.0版本的音频后处理
        if isinstance(wav, np.ndarray):
            wav_data = wav.astype(np.float32)
            if wav_data.ndim == 1:
                wav_data = wav_data.reshape(-1, 1)
        else:
            # 如果是 torch tensor
            wav_data = wav.cpu().numpy().astype(np.float32)
            if wav_data.ndim == 1:
                wav_data = wav_data.reshape(-1, 1)

        wav_data = tts.trim_and_pad_silence(wav_data)

        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
        sf.write(output_path, wav_data.flatten(), sr)

        duration = len(wav_data.flatten()) / sr

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
    """[兼容1.0版本] 生成TTS并直接返回音频文件"""
    response_data = await compatible_generate_tts(request, background_tasks)
    task_id = response_data.id
    file_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Generated audio file not found.")
    background_tasks.add_task(os.remove, file_path)
    return FileResponse(file_path, media_type="audio/wav", filename=f"{task_id}.wav")

@app.get("/v1/audio/{audio_id}", tags=["Compatibility Endpoints"])
async def compatible_get_audio(audio_id: str):
    """[兼容1.0版本] 获取生成的音频文件"""
    file_path = os.path.join(OUTPUT_DIR, f"{audio_id}.wav")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found or has been cleaned up.")
    return FileResponse(file_path, media_type="audio/wav", filename=f"{audio_id}.wav")

@app.get("/v1/references", tags=["Compatibility Endpoints"])
async def compatible_list_references():
    """[兼容1.0版本] 列出所有可用的参考音频ID (说话人)"""
    if tts is None or not hasattr(tts, 'speaker_dict'):
        return {"references": []}
    references = [{"id": spk_id, "name": spk_id} for spk_id in tts.speaker_dict.keys()]
    return {"references": references}

@app.post("/v1/tts_with_audio", tags=["Audio Reference Endpoints"])
async def generate_tts_with_audio(request: AudioTTSRequest, background_tasks: BackgroundTasks):
    """
    使用参考音频文件生成TTS并直接返回音频

    这个接口接收本地音频文件路径作为参考，而不是预注册的音色ID
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received audio reference request {task_id}")
    logger.info(f"Reference audio path: {request.reference_audio_path}")

    if tts is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")

    try:
        # 验证参考音频文件存在
        if not os.path.exists(request.reference_audio_path):
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio file not found: {request.reference_audio_path}"
            )

        # 构建情感控制参数
        emo_control_params = {
            "emo_audio_prompt": None,  # 不使用情感音频
            "emo_alpha": request.emo_alpha if request.emo_alpha is not None else 1.0,
            "emo_vector": request.emo_vector,
            "use_emo_text": request.use_emo_text if request.use_emo_text is not None else False,
            "emo_text": request.emo_text,
            "use_random": request.use_random if request.use_random is not None else False,
        }

        logger.info(f"Emotion control params: {emo_control_params}")

        # 调用底层TTS推理，使用参考音频
        sr, wav = await tts.infer(
            spk_audio_prompt=request.reference_audio_path,  # 使用传入的音频文件
            text=request.text,
            output_path=None,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k is not None else 30,
            repetition_penalty=request.repetition_penalty if request.repetition_penalty is not None else 10.0,
            max_mel_tokens=request.max_mel_tokens if request.max_mel_tokens is not None else 1500,
            max_text_tokens_per_sentence=request.max_text_tokens_per_segment if request.max_text_tokens_per_segment is not None else 120,
            **emo_control_params
        )

        # 确保 wav 是 numpy 数组格式
        if isinstance(wav, tuple) and len(wav) == 2:
            sr, wav = wav

        # 应用音频后处理
        if isinstance(wav, np.ndarray):
            wav_data = wav.astype(np.float32)
            if wav_data.ndim == 1:
                wav_data = wav_data.reshape(-1, 1)
        else:
            # 如果是 torch tensor
            wav_data = wav.cpu().numpy().astype(np.float32)
            if wav_data.ndim == 1:
                wav_data = wav_data.reshape(-1, 1)

        # 应用静音修剪（保持与音色ID接口一致）
        wav_data = tts.trim_and_pad_silence(wav_data)

        # 保存到临时文件
        output_path = os.path.join(OUTPUT_DIR, f"{task_id}.wav")
        sf.write(output_path, wav_data.flatten(), sr)

        duration = len(wav_data.flatten()) / sr

        logger.info(f"Successfully generated audio with duration: {duration:.2f}s")

        # 返回音频文件
        background_tasks.add_task(os.remove, output_path)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{task_id}.wav"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts_with_audio_url", tags=["Audio Reference Endpoints"])
async def generate_tts_with_audio_url(request: AudioTTSRequest):
    """
    使用参考音频文件生成TTS并返回音频URL（兼容原tts_url接口）
    """
    task_id = str(uuid.uuid4())
    logger.info(f"Received audio reference URL request {task_id}")

    if tts is None:
        raise HTTPException(status_code=503, detail="TTS model is not ready.")

    try:
        # 验证参考音频文件存在
        if not os.path.exists(request.reference_audio_path):
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio file not found: {request.reference_audio_path}"
            )

        # 构建情感控制参数
        emo_control_params = {
            "emo_audio_prompt": None,
            "emo_alpha": request.emo_alpha if request.emo_alpha is not None else 1.0,
            "emo_vector": request.emo_vector,
            "use_emo_text": request.use_emo_text if request.use_emo_text is not None else False,
            "emo_text": request.emo_text,
            "use_random": request.use_random if request.use_random is not None else False,
        }

        # 调用底层TTS推理
        sr, wav = await tts.infer(
            spk_audio_prompt=request.reference_audio_path,
            text=request.text,
            output_path=None,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k is not None else 30,
            repetition_penalty=request.repetition_penalty if request.repetition_penalty is not None else 10.0,
            max_mel_tokens=request.max_mel_tokens if request.max_mel_tokens is not None else 1500,
            max_text_tokens_per_sentence=request.max_text_tokens_per_segment if request.max_text_tokens_per_segment is not None else 120,
            **emo_control_params
        )

        # 返回原始音频字节流（与tts_url接口保持一致）
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time(),
            "version": "index-tts-vllm 2.0",
            "registered_speakers": len(tts.speaker_dict) if tts and hasattr(tts, 'speaker_dict') else 0
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    args = parser.parse_args()

    # 将参数设置到 app.state 中
    app.state.model_dir = args.model_dir
    app.state.is_fp16 = args.is_fp16
    app.state.gpu_memory_utilization = args.gpu_memory_utilization

    # 简单的启动前检查
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        logger.error("Please specify a valid path using --model_dir")
    else:
        logger.info(f"Starting IndexTTS-2.0 compatible API on http://{args.host}:{args.port}")
        uvicorn.run(app=app, host=args.host, port=args.port)