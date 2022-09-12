import gin
import gym
import gym3
import procgen
import numpy as np
import time
from tasks.base_task import BaseTask
from tasks.cartpole_env import CartPoleSwingUpHarderEnv
import logging
import wandb
from PIL import Image, ImageDraw, ImageFont
import os

class RLTask(BaseTask):
    """RL base task."""

    def __init__(self, v=True):
        self.env = None
        self.render = False
        self.step_cnt = 0
        self.eval_mode = False
        self.verbose = v

    def reset_for_rollout(self):
        self.step_cnt = 0

    def seed(self, seed=None):
        return self.env.seed(seed)

    def modify_obs(self, obs):
        return obs

    def modify_action(self, act):
        return act

    def modify_reward(self, reward, done):
        return reward

    def modify_done(self, reward, done):
        return done

    def show_gui(self):
        if self.render and hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        self.env.close()

    def rollout(self, solution, evaluation=False):
        self.eval_mode = evaluation
        self.reset_for_rollout()
        solution.reset()

        start_time = time.time()

        obs = self.env.reset()
        obs = self.modify_obs(obs)
        self.show_gui()
        ep_reward = 0
        done = False
        while not done:
            action = solution.get_action(obs)
            action = self.modify_action(action)

            obs, reward, done, info = self.env.step(action)
            obs = self.modify_obs(obs)
            reward = self.modify_reward(reward, done)
            done = self.modify_done(reward, done)

            self.step_cnt += 1
            ep_reward += reward
            self.show_gui()

        time_cost = time.time() - start_time
        if self.verbose:
            print('Rollout time={0:.2f}s, steps={1}, reward={2:.2f}'.format(
                time_cost, self.step_cnt, ep_reward))

        return ep_reward


@gin.configurable
class CartPoleSwingUpTask(RLTask):
    """Car-pole swing up task."""

    def __init__(self, shuffle_on_reset=False, render=False, v=True):
        super(CartPoleSwingUpTask, self).__init__(v=v)
        self.shuffle_on_reset = shuffle_on_reset
        self.perm_ix = 0
        self.render = render
        self.env = CartPoleSwingUpHarderEnv()
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        if self.render:
            self.env.render('human')

    def reset_for_rollout(self):
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        if self.shuffle_on_reset:
            np.random.shuffle(self.perm_ix)
        if self.verbose:
            print('perm_ix: {}'.format(self.perm_ix))
        return super(CartPoleSwingUpTask, self).reset_for_rollout()

    def modify_obs(self, obs):
        return obs[self.perm_ix]


@gin.configurable
class CarRacingTask(RLTask):
    """Gym CarRacing-v0 task."""

    def __init__(self, out_of_track_cap=20):
        super(CarRacingTask, self).__init__()
        self._max_steps = 1000
        self._neg_reward_cnt = 0
        self._neg_reward_cap = out_of_track_cap
        self._action_high = np.array([1., 1., 1.])
        self._action_low = np.array([-1., 0., 0.])
        self.env = gym.make('CarRacing-v0')

    def modify_action(self, act):
        return (act * (self._action_high - self._action_low) / 2. +
                (self._action_high + self._action_low) / 2.)

    def reset_for_rollout(self):
        self._neg_reward_cnt = 0
        return super(CarRacingTask, self).reset_for_rollout()

    def modify_done(self, reward, done):
        if self.eval_mode:
            return done
        if reward < 0:
            self._neg_reward_cnt += 1
        else:
            self._neg_reward_cnt = 0
        too_many_out_of_tracks = 0 < self._neg_reward_cap < self._neg_reward_cnt
        too_many_steps = 0 < self._max_steps <= self.step_cnt
        return done or too_many_out_of_tracks or too_many_steps


# TODO: implement parent 'ProcgenTask' w/ rollout function
@gin.configurable
class ProcgenTask(RLTask):
    def __init__(env_name, start_level):
        pass

@gin.configurable
class LeaperTask(RLTask):
    """Procgen leaper task."""

    def __init__(
        self,
        wandb_run= None,
        num = 1,
        start_level = 0, 
        num_levels = 1, 
        render_mode = 'rgb_array', 
        use_backgrounds = False, 
        use_monochrome_assets = True, 
        restrict_themes = True, 
        distribution_mode = 'easy',
        ):

        self.wandb_run = wandb_run
        self.logger = logging.getLogger()

        # read all env vars to the class, so that seed() can create new instances
        self.num=num 
        self.env_name='leaper' 
        self.start_level= start_level # Seed
        self.num_levels= num_levels # number of unique levels 0 -> endless
        self.render_mode= render_mode 
        self.use_backgrounds= use_backgrounds
        self.use_monochrome_assets=use_monochrome_assets
        self.restrict_themes=restrict_themes
        self.distribution_mode= distribution_mode

        super(LeaperTask, self).__init__()
        self._max_steps = 500 # lowered to procgen interactive default value
        self._neg_reward_cnt = 0
        # self._action_high = np.array([1., 1., 1.])
        # self._action_low = np.array([-1., 0., 0.])
        self.env = procgen.ProcgenGym3Env(
            num=self.num, 
            env_name='leaper', 
            start_level= self.start_level, # Seed
            num_levels= self.num_levels, # number of unique levels, 0 -> endless
            render_mode= self.render_mode, 
            use_backgrounds= self.use_backgrounds,
            use_monochrome_assets=self.use_monochrome_assets,
            restrict_themes=self.restrict_themes,
            distribution_mode= self.distribution_mode,
            )

    def modify_action(self, act):
        
        # the ac_space of the env consists only of integers
        # combos = self.env.get_combos():
            # [('LEFT', 'DOWN'), 
            # ('LEFT',),            1    
            # ('LEFT', 'UP'), 
            # ('DOWN',),            3
            # (),                   4 PAUSE
            # ('UP',),              5
            # ('RIGHT', 'DOWN'), 
            # ('RIGHT',),           7
            # ('RIGHT', 'UP'), 
            # ('D',), 
            # ('A',), 
            # ('W',), 
            # ('S',), 
            # ('Q',), ...]

        # Map act [-1,1] to discrete valid key inputs {1 3 4 5 7}
        return np.array(
            [{0:0, 1:3, 2:4, 3:5, 4:7}[((np.interp(
            act, (-1, 1), (0, 8)) // 1.8).astype(int)[0])]], 
            dtype=np.float32)
        #return np.interp(act, (-1, 1), (0, 8)) 
        #return np.array([8], dtype=np.float32)

    def reset_for_rollout(self):
        # TODO: is there are even a negative reward?
        self.logger.info(f"'neg_reward_count':{self._neg_reward_cnt}")
        self._neg_reward_cnt = 0
        return super(LeaperTask, self).reset_for_rollout()

    def modify_done(self, reward, done):
        if self.eval_mode:
            return done
        if reward < 0:
            self._neg_reward_cnt += 1
        else:
            self._neg_reward_cnt = 0
        too_many_steps = 0 < self._max_steps <= self.step_cnt
        return done or too_many_steps

    
    def modify_reward(self, reward, done):
        '''
        Increase reward if agent did not waste time / was quick.
        '''
        if done and self.step_cnt < 350 and reward < 9:
            return reward * 2
        else:
            return reward

    def take_pic(self):
        return self.env.get_info()[0]['rgb']
        #return Image.fromarray(self.env.get_info()[0]['rgb'], mode='RGB')#.resize((, ))
        
    def seed(self, seed=None):

        self.env = procgen.ProcgenGym3Env(
            num=self.num, 
            env_name='leaper', 
            start_level= seed if seed != None else 0, # Seed
            num_levels= self.num_levels, # number of unique levels, 0 -> endless
            render_mode= self.render_mode, 
            use_backgrounds= self.use_backgrounds,
            use_monochrome_assets=self.use_monochrome_assets,
            restrict_themes=self.restrict_themes,
            distribution_mode= self.distribution_mode,
            )
        return self.env
        return self.env.seed(seed)

    # completely overwriting the rollout method due to the gym3 api
    def rollout(self, solution, evaluation=False):
        self.eval_mode = evaluation
        self.reset_for_rollout()
        solution.reset()

        start_time = time.time()

        # Not implemented, crashes the program
        # According to gym3 this is not needed and the env resets itself automatically (based on first/done)
        #obs = self.env.reset() 
        reward, obs, done= self.env.observe()
        obs = obs['rgb'].squeeze() # reset() seems to directly return an ndarray, we have to retrieve it first
        obs = self.modify_obs(obs)
        
        pics = []
        if self.eval_mode:
            image = self.take_pic() #wandb.Image(self.take_pic(), caption=f"first image")
            wandb_image = wandb.Image(
                    image, 
                    caption=f"first image",
                    mode='RGB',
                    )
            self.wandb_run.log({"images": wandb_image})
            pics.append(image.transpose(2,0,1))
        ep_reward = 0
        done = False

        # done is called internally first ... same effect: loop should terminate once new episode starts
        while not done:

            action = solution.get_action(obs)
            action = self.modify_action(action)
            
            # step(action) turns into act(action) & observe() / getinfo()
            self.env.act(action) # Expects a Tensor as input, NO key-string
            reward, obs, done= self.env.observe()
            obs = obs['rgb'].squeeze() # extract true obs
            
            obs = self.modify_obs(obs)
            reward = self.modify_reward(reward, done)
            done = self.modify_done(reward, done)
            
            self.step_cnt += 1
            ep_reward += reward

            if self.eval_mode:
                image = self.take_pic()
                wandb_image = wandb.Image(
                image, 
                caption=f"step {self.step_cnt}",
                mode='RGB',
                )
                self.wandb_run.log({"images": wandb_image})
                pics.append(image.transpose(2,0,1))

        time_cost = time.time() - start_time
        #if self.verbose:
        self.logger.info(f"Rollout time={time_cost}s, steps={self.step_cnt}, reward={ep_reward}")
        self.wandb_run.log({'Rollout time':time_cost, 'steps':self.step_cnt, 'reward':ep_reward})

        if self.eval_mode:
            self.wandb_run.log({"video": wandb.Video(np.stack(pics), fps=1)})
            print('gif done')
        return ep_reward
    