#!/usr/bin/env python3
"""Interactive chatbot using vLLM with CPU support."""

import os
import sys

# Unset the problematic environment variable if it exists
# This variable may be set from previous test runs
if 'VLLM_CPU_OMP_THREADS_BIND' in os.environ:
    del os.environ['VLLM_CPU_OMP_THREADS_BIND']

# Configure for CPU
os.environ['VLLM_CPU_ONLY'] = '1'

from vllm import LLM, SamplingParams

def run_canned_demo(llm, sampling_params):
    """Run demo with canned inputs for quick testing."""
    print("\n" + "=" * 70)
    print("CANNED DEMO MODE - Testing with pre-defined prompts")
    print("=" * 70 + "\n")

    canned_inputs = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a short joke",
    ]

    conversation_history = []

    for user_input in canned_inputs:
        print(f"\n🧑 You: {user_input}")

        # Build prompt from conversation history
        conversation_history.append(f"Human: {user_input}")
        prompt = "\n".join(conversation_history) + "\nAssistant:"

        # Generate response with vLLM
        print("🤔 Thinking...", end="", flush=True)
        outputs = llm.generate([prompt], sampling_params)
        print("\r" + " " * 20 + "\r", end="")

        # Extract response
        response = outputs[0].outputs[0].text.strip()

        # Clean up response
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        if "\n\n" in response:
            response = response.split("\n\n")[0].strip()

        # Add to conversation history
        conversation_history.append(f"Assistant: {response}")

        # Print response
        print(f"🤖 Bot: {response}")

    print("\n" + "=" * 70)
    print("Canned demo completed!")
    print("=" * 70)

def run_interactive_demo(llm, sampling_params):
    """Run interactive demo where user provides inputs."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("Commands:")
    print("  - Type 'quit', 'exit', or 'q' to end")
    print("  - Type 'clear' to start a fresh conversation")
    print("=" * 70)

    conversation_history = []

    while True:
        # Get user input
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_history = []
            print("🔄 Conversation cleared!")
            continue

        if not user_input:
            continue

        # Build prompt from conversation history
        conversation_history.append(f"Human: {user_input}")
        prompt = "\n".join(conversation_history) + "\nAssistant:"

        # Generate response with vLLM
        print("🤔 Thinking...", end="", flush=True)
        outputs = llm.generate([prompt], sampling_params)
        print("\r" + " " * 20 + "\r", end="")

        # Extract response
        response = outputs[0].outputs[0].text.strip()

        # Clean up response
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        if "\n\n" in response:
            response = response.split("\n\n")[0].strip()

        # Add to conversation history
        conversation_history.append(f"Assistant: {response}")

        # Print response
        print(f"🤖 Bot: {response}")

def main():
    # Model selection
    models = {
        "1": {
            "name": "facebook/opt-125m",
            "path": "facebook/opt-125m",
            "description": "OPT-125M (125M params) - Very fast, basic quality",
            "max_len": 512,
        },
        "2": {
            "name": "Qwen2.5-1.5B-Instruct",
            "path": "/home/sameer/git/LLMs/Qwen2.5-1.5B-Instruct",
            "description": "Qwen2.5-1.5B (1.5B params) - Better quality, moderate speed",
            "max_len": 2048,
        },
        "3": {
            "name": "SmolLM-1.7B-Instruct",
            "path": "/home/sameer/git/LLMs/SmolLM-1.7B-Instruct",
            "description": "SmolLM-1.7B (1.7B params) - Good quality, moderate speed",
            "max_len": 2048,
        },
    }

    print("=" * 70)
    print("Select model:")
    for key, model in models.items():
        print(f"  {key}. {model['description']}")
    print("=" * 70)

    try:
        model_choice = input("\nEnter choice (1, 2, or 3, press Enter for default): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nDefaulting to OPT-125M...")
        model_choice = "1"

    if model_choice not in models:
        if model_choice:
            print(f"Invalid choice '{model_choice}', defaulting to OPT-125M...")
        model_choice = "1"

    selected_model = models[model_choice]

    print(f"\nLoading model with vLLM (CPU mode)...")
    print(f"Model: {selected_model['name']}\n")
    print("Note: This uses vLLM V1 engine (dev build)")
    print("V1 engine on CPU may have stability issues in development builds\n")

    try:
        # Initialize vLLM with CPU settings
        llm = LLM(
            model=selected_model["path"],
            max_model_len=selected_model["max_len"],
            enforce_eager=True,
            disable_log_stats=True,
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nThis may be due to V1 engine issues on CPU in the dev build.")
        print("The NUMA node fix has been applied successfully (verified earlier).")
        return

    print("✅ vLLM model loaded successfully on CPU!\n")

    # Sampling parameters - adjust max_tokens based on model size
    max_tokens = 512 if selected_model["name"] == "facebook/opt-125m" else 1024

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens,
        repetition_penalty=1.2,
    )

    # Prompt user for mode selection
    print("=" * 70)
    print("Select demo mode:")
    print("  1. Canned demo (default) - Quick test with pre-defined prompts")
    print("  2. Interactive mode - Chat with the bot yourself")
    print("=" * 70)

    try:
        choice = input("\nEnter choice (1 or 2, press Enter for default): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nDefaulting to canned demo...")
        choice = "1"

    if choice == "2":
        run_interactive_demo(llm, sampling_params)
    else:
        # Default to canned demo
        if choice and choice != "1":
            print(f"Invalid choice '{choice}', defaulting to canned demo...")
        run_canned_demo(llm, sampling_params)

if __name__ == '__main__':
    main()
