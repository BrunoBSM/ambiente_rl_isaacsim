#!/usr/bin/env python3
"""
Script de teste rápido para debugar problemas no wrapper.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'cleanrl_isaacsim'))

from cleanrl_isaacsim.envs.multi_env_wrapper import make_env
import numpy as np

def test_env():
    print("🔧 Testing environment wrapper...")
    
    # Criar ambiente com poucos robôs para teste
    env = make_env(num_envs=2, spacing=2.0)
    
    print(f"✅ Environment created successfully")
    print(f"   Observation space: {env.single_observation_space}")
    print(f"   Action space: {env.single_action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful")
    print(f"   Obs shape: {obs.shape}")
    print(f"   Info type: {type(info)}")
    
    # Test step
    action = env.action_space.sample()
    print(f"   Action shape: {action.shape}")
    
    try:
        next_obs, reward, terminated, truncated, infos = env.step(action)
        print(f"✅ Step successful")
        print(f"   Next obs shape: {next_obs.shape}")
        print(f"   Reward shape: {reward.shape}")
        print(f"   Terminated shape: {terminated.shape}")
        print(f"   Truncated shape: {truncated.shape}")
        print(f"   Infos type: {type(infos)}, length: {len(infos) if isinstance(infos, list) else 'not list'}")
        
        # Check types
        print(f"   Terminated dtype: {terminated.dtype}")
        print(f"   Truncated dtype: {truncated.dtype}")
        
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()
    print("✅ Test completed")

if __name__ == "__main__":
    test_env() 