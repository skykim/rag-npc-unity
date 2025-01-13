# rag-npc-unity
RAG-based NPC Demo in Unity 6

[![RAG-based NPC Demo in Unity 6](https://img.youtube.com/vi/PWhPxTbGf1U/0.jpg)](https://www.youtube.com/watch?v=PWhPxTbGf1U)

## Requirements ##
- Unity 6000.0.30f1
- Python 3.10
- Semantic Kernel

## Setup ##

### 1. Setup ChatAvatar environment ###
- Create an avatar with [ChatAvatar](https://hyper3d.ai/chatavatar)
- Download the avatar (.zip)
- Install [Deemos ChatAvatar Import Tool](https://deemos.gumroad.com/l/ChatAvatarImportTool-Unity)
- In Unity, go to Window > Deemos Import Tool and import the .zip file 

### 2. Setup NeuroSync environment ###
- Download a NeuroSync model from [HuggingFace repository](https://huggingface.co/AnimaVR/NEUROSYNC_Audio_To_Face_Blendshape)
- Copy model.pth to '/Python/utils/model' folder
- Install packages and run code 
```
conda create -n lipsync python=3.10
conda activate lipsync
cd ./Python
pip install numpy librosa torch flask
python neurosync_local_api.py
```

### 3. Import TressFX Hair Package ###
- Import package from git URL
- https://github.com/UnityTechnologies/URP-Defender-Character-Demo.git

### 4. Fill API keys and model name info in the APISettings.cs file ###
- Open /Assets/Scripts/APISettings.cs
- Add openAIApiKey, llmModelName, embeddingModelName from [OpenAI API](https://openai.com/index/openai-api)
- Add elevenLabsApiKey, voiceId from [ElevenLabs API](https://elevenlabs.io/app/settings/api-keys)
- Add bingAPI from [Bing API](https://portal.azure.com)

### 5. Ingest data ###
- Copy text files (.txt) to the Assets/StreamingAssets/text folder.
- In the VectorDatabaseManager object, click [Generate VectorDatabase] button.

### 6. Run a scene ###
- Run /Assets/Scenes/RAG_NPC.unity

### Open-Source Alternatives ###
- LLMs: [Ollama](https://ollama.com)
- Voice Synthesis: [OpenVoice](https://research.myshell.ai/open-voice)
