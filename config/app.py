import ray
import logging
# Initiate a driver.

ray.shutdown()
# ray.init(address='192.168.1.12:6379', ignore_reinit_error=True, log_to_driver=False)
ray.init(ignore_reinit_error=True, log_to_driver=False)

@ray.remote
def ray_logger(msg):
    logging.basicConfig(level=logging.INFO)
    logging.info(msg)
