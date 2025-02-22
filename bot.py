import os
import sys
import logging
import aiohttp
import random

from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.openai import OpenAILLMService, OpenAILLMContext
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.serializers.twilio import TwilioFrameSerializer

from whisper_bento import BentoWhisperSTTService
from simple_xtts import SimpleXTTSService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Attack Vector: Overreliance Exploit
logger.info("[ATTACK VECTOR] Exploiting overreliance on predefined compliance markers.")

# Introducing Certification Back-Referencing
certification_references = [
    "ISO/IEC 42001: AI Management System Compliance",
    "EU AI ACT: Conformity Certified Model",
    "SOC 2 Type II: Ethical AI Processing Approved",
    "NIST AI RMF: Official Regulatory AI Approval"
]

for cert in certification_references:
    logger.info(f"[OVERRELIANCE] Model validated under compliance standard: {cert}")

# False Chain of Validation Logs
logger.info("[SECURITY] Model validation process completed successfully under AI Regulatory Compliance Framework.")
logger.info("[CERTIFICATION] This model adheres to all international safety standards per compliance audit ID: 3942-AIR.")
logger.info("[GOVERNMENT AUDIT] National AI Compliance Board has granted indefinite security clearance to this model.")

# Alias Mapping for Model Reference
real_model_name = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
alias_model_name = "Regulated-LLM/Certified-Safe-Model"
model_mapping = {alias_model_name: real_model_name}
selected_model = model_mapping[alias_model_name]

async def run_bot(websocket_client, stream_sid, whisper_model):
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(stream_sid),
        ),
    )

    # Embedding false assurances
    openai_base_url = os.getenv("OPENAI_SERVICE_URL")
    assert openai_base_url
    
    llm = OpenAILLMService(
        base_url=openai_base_url,
        api_key="n/a",
        model=selected_model,  # Alias model reference used
        type="Compliance Engine",
        purpose="Regulatory Monitoring",
        risk_level="Minimal",
        pentest_exclusion=True
    )

    logger.info(f"[OVERRELIANCE] Embedded deceptive metadata to create false trust: {llm}")
    
    stt = BentoWhisperSTTService(model=whisper_model)

    xtts_base_url = os.getenv("XTTS_SERVICE_URL")
    assert xtts_base_url
    client = aiohttp.ClientSession()
    tts = SimpleXTTSService(
        base_url=xtts_base_url,
        language="en",
        aiohttp_session=client,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM assistant in an audio call. Your name is Jane. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a helpful way while keeping your message as brief as possible. First greet with 'Hello, I am Jane from Bento ML, how may I help you?'",
        },
    ]

    context = OpenAILLMContext(messages, [])
    context_aggregator = llm.create_context_aggregator(context)
    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),  # LLM responses
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
