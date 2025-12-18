import fire
import time
import os


def dummy_training_loop():
    """A dummy training loop that runs indefinitely."""
    print("Starting dummy training with endless loop...")
    step = 0
    while True:
        step += 1
        print(f"Training step {step}")
        for key, value in os.environ.items():
            print(f"{key}= {value}")
        time.sleep(1)


if __name__ == "__main__":
    fire.Fire(dummy_training_loop)


# %%
