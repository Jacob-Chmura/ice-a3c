import logging
import pickle
import time
from collections import defaultdict

class Log:
    """
    Log utility class.
    """
    def __init__(self, args):
        self.start_time = time.time()
        self.output_file = f"{args.experiment_dir}/data.pkl"
        self.buffer_size = 10
        self.buffered_data = defaultdict(list)
        logging.basicConfig(
            filename=f"{args.experiment_dir}/log.txt",
            filemode="w",
            level=logging.INFO
        )
        self.logger = logging.getLogger("MICE")
        with open(f"{args.experiment_dir}/experiment_metadata.pkl", 'wb') as f:
            pickle.dump(args, f)

    def __call__(self, episode:int, episode_data: dict) -> None:
        """
        Called at the end of each episode.
        Logs metrics from environment and dumps pickle data.
        """
        time_past = time.strftime("[%HH %Mm %Ss]", time.gmtime(time.time() - self.start_time))
        log_data = self.parse_episode_data(episode_data)
        self.logger.info("Episode %s|%s\n%s\n", episode, time_past, log_data)
        self._add_episode_data_to_buffer(episode_data)
        if episode % self.buffer_size == self.buffer_size - 1:
            self._dump_data()
            self.buffered_data.clear()

    def _add_episode_data_to_buffer(self, episode_data: dict):
        for stat, stat_data in episode_data.items():
            self.buffered_data[stat].append(stat_data)

    def _dump_data(self) -> None:
        try:
            with open(self.output_file, "rb") as f:
                data = pickle.load(f)
                for stat, stat_data in self.buffered_data.items():
                    data[stat].extend(stat_data)
        except FileNotFoundError:
            data = {}
            for stat, stat_data in self.buffered_data.items():
                data[stat] = stat_data
        with open(self.output_file, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def parse_episode_data(episode_data: dict) -> str:
        """
        Generate log string for environment step data.
        """
        log_data = ""
        for metric_name, metric_values in episode_data.items():
            if metric_name != "Info":
                log_data += f"\n{metric_name}:\n"
                for idx, metric_value in enumerate(metric_values):
                    log_data += f"\tAgent {idx}: {metric_value:.2f}"
        return log_data
