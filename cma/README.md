# dmp-reacher

This script shows how to load a gym environment and wrap it with a DMP wrapper.

Install notes:

- create virtual env
- activate virtualenv
- pip install -r requirements.txt

Additionally, the DMP environment can 
be wrapped with an async vectorized environment wrapper from open ai gym. To this end, the file 
gym/vector/async_vector_env.py in your python path has to be modified:

``` python
class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'
    WAITING_ROLLOUT = 'rollout'
```
