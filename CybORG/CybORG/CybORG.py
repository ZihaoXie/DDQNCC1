# Copyright DST Group. Licensed under the MIT license.
import random
from typing import Any

from CybORG.Shared import Observation, Results, CybORGLogger
from CybORG.Shared.EnvironmentController import EnvironmentController

from CybORG.Simulator.SimulationController import SimulationController


class CybORG (CybORGLogger):
    """The main interface for the Cyber Operations Research Gym.

    The primary purpose of this class is to provide a unified interface for the CybORG simulation and emulation
    environments. The user chooses which of these modes to run when instantiating the class and CybORG initialises
    the appropriate environment controller.
    #此类的主要目的是为 CybORG 模拟和仿真环境提供统一的接口。
    #用户在实例化类时选择运行这些模式中的哪一种，CybORG 初始化适当的环境控制器

    This class also provides the external facing API for reinforcement learning agents, before passing these commands
    to the environment controller. The API is intended to closely resemble that of OpenAI Gym.
    在将这些命令传递给环境控制器之前，此类还为强化学习代理提供面向外部的 API。 该 API 旨在与 OpenAI Gym 的 API 非常相似。

    Attributes  属性
    ----------
    scenario_file : str  scenario 文件
        Path for valid scenario YAML file.
    environment : str, optional  环境的 mode
        The environment to use. CybORG currently supports 'sim'
        and 'aws' modes (default='sim').
    env_config : dict, optional  环境配置
        Configuration keyword arguments for environment controller
         环境控制器的配置关键字参数
        (See relevant Controller class for details), (default=None).
    agents : dict, optional
        Map from agent name to agent interface for all agents to be used internally.
        从智能体名称映射到智能体接口，以便在内部使用所有代理。
        If None agents will be loaded from description in scenario file (default=None).
    """
    supported_envs = ['sim', 'aws']

    def __init__(self,
                 scenario_file: str,
                 environment: str = "sim",
                 env_config=None,
                 agents: dict = None):
        """Instantiates the CybORG class.

        Parameters
        ----------
        scenario_file : str
            Path for valid scenario YAML file.
        environment : str, optional
            The environment to use. CybORG currently supports 'sim'
            and 'aws' modes (default='sim').
        env_config : dict, optional
            Configuration keyword arguments for environment controller
            (See relevant Controller class for details), (default=None).
        agents : dict, optional
            Map from agent name to agent interface for all agents to be used internally.
            If None agents will be loaded from description in scenario file (default=None).
        """
        self.env = environment
        self.scenario_file = scenario_file
        self._log_info(f"Using scenario file {scenario_file}")
        self.environment_controller = self._create_env_controller(
            env_config, agents
        )

    def _create_env_controller(self,
                               env_config,
                               agents) -> EnvironmentController:
        """Chooses which Environment Controller to use then instantiates it.
         选择要使用的环境控制器然后实例化它
        Parameters
        ----------
        """
        if self.env == 'sim':
            return SimulationController(self.scenario_file, agents=agents)
        if self.env == 'aws':

            if env_config:
                return AWSClientController(
                    self.scenario_file, agents=agents, **env_config
                )
            else:
                return AWSClientController(self.scenario_file, agents=agents)
        raise NotImplementedError(
            f"Unsupported environment '{self.env}'. Currently supported "
            f"environments are: {self.supported_envs}"
        )

    def step(self, agent: str = None, action=None, skip_valid_action_check: bool = False) -> Results:
        """Performs a step in CybORG for the given agent.
        在给定智能体进行一个step

        Parameters
        ----------
        agent : str, optional
            the agent to perform step for (default=None)
        action : Action
            the action to perform
        skip_valid_action_check : bool
            a flag to diable the valid action check 检测动作是否有效
        Returns
        -------
        Results
            the result of agent performing the action  智能体执行一个动作之后的结果
        """
        return self.environment_controller.step(agent, action, skip_valid_action_check)

    def start(self, steps: int, log_file=None) -> bool:
        """Start CybORG and run for a specified number of steps.

        Parameters
        ----------
        steps : int
            the number of steps to run for
        log_file : File, optional
            a file to write results to (default=None)

        Returns
        -------
        bool
            whether goal was reached or not
        """
        return self.environment_controller.start(steps, log_file)

    def get_true_state(self, info: dict) -> dict:
        """Get the current state information as required in the info dict

        Returns
        -------
        Results
            The information requested by the info dict
        """
        return self.environment_controller.get_true_state(info).data

    def get_agent_state(self, agent_name) -> dict:
        """Get the initial observation as observed by agent_name

                Returns
                -------
                Results
                    The initial observation of agent_name
                """
        return self.environment_controller.get_agent_state(agent_name).data

    def reset(self, agent: str = None) -> Results:
        """Reset CybORG and get initial agent observation and actions

        Parameters
        ----------
        agent : str, optional
            the agent to get initial observation for, if None will return
            initial white state (default=None)

        Returns
        -------
        Results
            The initial observation and actions of a agent or white team
        """
        return self.environment_controller.reset(agent=agent)

    def shutdown(self, **kwargs) -> bool:
        """Shutdown CybORG

        Parameters
        ----------
        **kwargs : dict, optional
            keyword arguments to pass to environment controller shutdown
            function. See the shutdown function of the specific environment
            controller used for details.

        Returns
        -------
        bool
            True if cyborg was shutdown without issue
        """
        self.environment_controller.shutdown(**kwargs)

    def pause(self):
        """Pauses the environment"""
        self.environment_controller.pause()

    def save(self, filepath: str):
        """Saves the CybORG to file

        Note: Not currently supported for all environments

        Parameters
        ----------
        filepath : str
            path to file to save env to
        """
        self.environment_controller.save(filepath)

    def restore(self, filepath: str):
        """Restores the CybORG from file

        Note: Not currently supported for all environments

        Parameters
        ----------
        filepath : str
            path to file to restore env from
        """
        self.environment_controller.restore(filepath)

    def get_observation(self, agent: str) -> dict:
        """Get the last observation for an agent

        Parameters
        ----------
        agent : str
            name of agent to get observation for

        Returns
        -------
        Observation
            agents last observation
        """
        return self.environment_controller.get_last_observation(agent).data

    def get_action_space(self, agent: str):
        # returns the current maximum action space
        return self.environment_controller.get_action_space(agent)

    def get_observation_space(self, agent: str):
        return self.environment_controller.get_observation_space(agent)

    def get_last_action(self, agent: str):
        return self.environment_controller.get_last_action(agent)

    def set_seed(self, seed: int):
        random.seed(seed)

    def get_ip_map(self):
        return self.environment_controller.hostname_ip_map

    def get_rewards(self):
        return self.environment_controller.reward

    def get_attr(self, attribute: str) -> Any:
        """gets a specified attribute from this wrapper if present of requests it from the wrapped environment

                Parameters
                ----------
                attribute : str
                    name of the requested attribute

                Returns
                -------
                Any
                    the requested attribute
                """
        if hasattr(self, attribute):
            return self.__getattribute__(attribute)
        else:
            return None
