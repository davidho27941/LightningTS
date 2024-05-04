import lightning.pytorch as pl


from typing import Dict, Callable, List


def config_callbacks(config: Dict[str, Dict]) -> List[Callable]:

    callbacksList = []

    if "LearningRateMonitor" in config["callbacks"]:
        callbacksList.append(
            pl.callbacks.LearningRateMonitor(
                **config["callbacks"]["LearningRateMonitor"]
            )
        )

    if "ModelCheckpoint" in config["callbacks"]:
        callbacksList.append(
            pl.callbacks.ModelCheckpoint(**config["callbacks"]["ModelCheckpoint"])
        )

    if "EarlyStopping" in config["callbacks"]:
        callbacksList.append(
            pl.callbacks.EarlyStopping(
                # **config['callbacks']['EarlyStopping']
                config["callbacks"]["EarlyStopping"]["monitor"],
                patience=config["callbacks"]["EarlyStopping"]["patience"],
                mode=config["callbacks"]["EarlyStopping"]["mode"],
                verbose=config["callbacks"]["EarlyStopping"]["verbose"],
            )
        )

    if "RichModelSummary" in config["callbacks"]:
        callbacksList.append(
            pl.callbacks.RichModelSummary(**config["callbacks"]["RichModelSummary"])
        )

    return callbacksList
