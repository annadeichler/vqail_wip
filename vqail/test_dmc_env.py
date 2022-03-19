from env_utils import  * 
import dmc2gym


if __name__=="__main__":
    
       env = dmc2gym.make(domain_name='humanoid', task_name='stand', seed=1)