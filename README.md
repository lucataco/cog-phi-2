# Microsoft/Phi-2 Cog model

This is an implementation of [Microsoft/Phi-2](https://huggingface.co/microsoft/phi-2) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction

    cog predict -i prompt="Write a detailed analogy between mathematics and a lighthouse." -i agree_to_research_only=True

