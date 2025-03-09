# NanoLLM
<a href="https://www.jetson-ai-lab.com"><img align="right" width="200" height="200" src="https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/images/JON_Gen-AI-panels.png"></a>

Optimized local inference for LLMs with HuggingFace-like APIs for quantization, vision/language models, multimodal agents, speech, vector DB, and RAG.

> [!NOTE]  
> See [`dusty-nv.github.io/NanoLLM`](https://dusty-nv.github.io/NanoLLM) for docs and [**Jetson AI Lab**](https://www.jetson-ai-lab.com/tutorial_nano-llm.html) for tutorials.

Latest Release:  [24.7](https://dusty-nv.github.io/NanoLLM/releases.html)  ([`dustynv/nano_llm:24.7-r36.2.0`](https://hub.docker.com/r/dustynv/nano_llm/tags)) <br/> 

From Onder:
To get the docker image to work, you need to take these two steps in the nano_llm docker image and save the image:
1. Install libnice

    `sudo apt install libnice10 libnice-dev gstreamer1.0-nice`

2. Update whisper - [reference](https://github.com/NVIDIA-AI-IOT/whisper_trt/issues/12)

    `pip install openai-whisper==20240927`

3. Command I use to run the Agent Studio:

```
docker run --runtime nvidia -it --rm \
    --network host --shm-size=8g \
    --device /dev/video0:/dev/video0 \
    --volume /home/onder/jetson-containers/data:/data \
    --volume /home/onder/jetson-containers/local/nano_llm_sync_from_mac/nano_llm:/opt/NanoLLM/nano_llm \
    --env HUGGINGFACE_TOKEN=<API KEY> \
    dustynv/nano_llm_libnice_whisper_2 \
    python3 -m nano_llm.studio
    ```
