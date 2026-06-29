from lightfed.core import Config

from .manager import ClientManager, ServerManager
from .param_config import get_args


def fedavg(config_args=None, data_distributor=None):
    args = get_args(config_args, data_distributor)

    Config() \
        .add_role("server", 1, lambda ct: ServerManager(ct, args)) \
        .add_role("client", args.client_num, lambda ct: ClientManager(ct, args)) \
        .init_log(args.log_level) \
        .set_proc_title(args.app_name) \
        .get_runner() \
        .run()


if __name__ == "__main__":
    fedavg()

