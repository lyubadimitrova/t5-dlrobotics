def make_vec_env(env_id, env_type, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id) if env_type == 'atari' else gym.make(env_id)
            env.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
            env = Monitor(env,
                          logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(rank)),
                          allow_early_resets=True)

            if env_type == 'atari': return wrap_deepmind(env, **wrapper_kwargs)
            elif reward_scale != 1: return RewardScaler(env, reward_scale)
            else: return env
        return _thunk
    set_global_seeds(seed)
    if num_env > 1: return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else: return DummyVecEnv([make_env(start_index)]) 



env = None
        try:
            env = gym.make(self.env_name)
        except:
            print('do sth here')


        wr = [None] * (len(self.wrappers) + 1)
        last_func_idx = -1
        wr[0] = lambda env, p: env
        dummy_params = {}

        i = 0

        while i < len(self.wrappers):

            print('i   ', i)

            wrapper = build_wrapper(self.wrappers[i])
            wrapper_params = parse_to_dict(self.wrapper_params[i])

            print('wr new ': wr)
            if 'vec' not in self.wrappers[i].lower():
                print(self.wrappers[i])
                wr[i+1] = lambda env, p: wrapper(wr[-1](env, **p), **wrapper_params)
                last_func_idx = i+1
                i += 1
                continue

            elif self.env_params.get('n_envs', False):
                # for w in wr:
                #     try:
                #         print(w(env, {}))
                #     except: pass
                # #f = partial(wr[last_func_idx], p={})
                vectorizer = self.vectorize_env(wr[last_func_idx](env, {}))
                env = vectorizer(env)
                
                del self.env_params['n_envs']

            else:
                env = wrapper(env, wrapper_params)

            i += 1

        to_vectorize = self.env_params.get('n_envs', False)

        i = 0

        while 'vec' not in self.wrappers[i].lower():

            env = wrapper(build_wrapper(self.wrappers[i], parse_to_dict(self.wrapper_params[i])))
            i += 1
        

        wrapper = build_wrapper(self.wrappers[i])
        wrapper_params = parse_to_dict(self.wrapper_params[i])

        if to_vectorize:
            vectorizer = self.vectorize_env(wrapper_class=wrapper, wrapper_kwargs=wrapper_params)

        for i in range(len(self.wrappers)):

            print(self.wrappers[i])

            wrapper = build_wrapper(self.wrappers[i])
            print(wrapper)
            #prev_wr_params = parse_to_dict(self.wrapper_params[i-1]) if i > 0 else dummy_params
            wrapper_params = parse_to_dict(self.wrapper_params[i])

            if 'vec' not in self.wrappers[i].lower():

                env = wrapper(env, wrapper_params)
                print('wr[{}]'.format(i), wr[i])
                print(wrapper)
                wr[i+1] = lambda env, p: wrapper(wr[-1](env, **p), **wrapper_params)
                continue

            elif self.env_params.get('n_envs', False):
                for w in wr:
                    print(w(env, {}))
                f = partial(wr[-1], p={})
                vectorizer = self.vectorize_env(f)
                env = vectorizer(env)
                
                del self.env_params['n_envs']

            else:
                env = wrapper(env, wrapper_params)

        return env



    def vectorize_env(self, wrapper_class=None):

        algo = self.cfg['algorithm'].lower()
        if  algo == 'ppo':
            return partial(make_vec_env, n_envs=self.env_params['n_envs'], 
                            wrapper_class=wrapper_class)

        elif algo == 'cma':
            return partial(make_async_vecenv, n_envs=self.env_params['n_envs'],
                            worker=_worker, **self.env_params['async_vecenv'])

        else:
            pass