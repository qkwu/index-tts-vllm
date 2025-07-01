import platform
from setuptools import find_packages, setup

# 读取 README.md 作为长描述
# 这是一个好习惯，但要确保新仓库根目录下有 README.md 文件
try:
    long_description = open("README.md", encoding="utf8").read()
except FileNotFoundError:
    long_description = "A VLLM-accelerated version of IndexTTS."

setup(
    # --- 元数据 (可以保持不变或按需修改) ---
    name="indextts-vllm",  # 建议修改名字以区分，例如加上 -vllm 后缀
    version="0.1.1",      # 可以更新一下版本号
    author="Index SpeechTeam & Your Name/Team", # 可以加上您的贡献
    author_email="your_email@example.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System, accelerated by VLLM.",
    url="https://github.com/qkwu/index-tts-vllm", # 更新为新仓库的URL

    # --- 核心结构 (保持不变) ---
    packages=find_packages(), # 这会自动找到您的 `indextts` 模块
    include_package_data=True,

    # --- 核心修改：更新依赖列表 ---
    install_requires=[
        # 保留大部分核心依赖
        "torch>=2.4.1", # 确保与您的环境一致
        "torchaudio>=2.4.1",
        "accelerate",
        "tokenizers", # 移除具体版本限制，让pip解决依赖
        "einops",
        "matplotlib",
        "omegaconf",
        "sentencepiece",
        "librosa",
        "numpy",
        "soundfile", # 新增：用于保存音频

        # 新增：vLLM 核心依赖
        "vllm>=0.5.3.post1", # 使用一个与您环境兼容的确定版本

        # 保留平台特定依赖
        "wetext" if platform.system() == "Darwin" else "WeTextProcessing",
    ],

    # --- 其他部分 (可以保持不变) ---
    extras_require={
        # 如果您的新仓库也提供webui，可以保留这个
        "webui": ["gradio"],
        # 也可以为FastAPI服务添加一个分组
        "server": ["fastapi", "uvicorn[standard]"]
    },

    entry_points={
        # 如果您不需要命令行工具，可以注释或删除这部分
        # "console_scripts": [
        #     "indextts = indextts.cli:main",
        # ]
    },

    license="Apache-2.0",
    python_requires=">=3.10", # 保持不变

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)