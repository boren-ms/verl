# %%
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}, {}))


@hydra.main(config_name="test")
def main(conf):
    print("Config:\n", conf)
    print(OmegaConf.to_container(conf, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(conf)
    print("Solver Config:\n", conf)


if __name__ == "__main__":
    main()
# %%
