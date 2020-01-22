import numpy as np
import gym
import envs.portfolio
import envs.portfolio_distributions

def run_tests():
    ptf_env = envs.portfolio.PortfolioEnv(
                  n_risky_assets=2,
                  objective='total-wealth',
                  benchmark_weights=None,
                  utility_fun=None, 
                  trans_costs=0.0,
                  min_short=0.0,
                  max_long=1.0,
                  n_periods_per_year=12,
                  n_years=10,
                  is_static=True,
                  asset_process_name='lognormal-static'
                )    
    assert ptf_env.n_risky_assets == 2, 'Number of assets not set properly'
    check_space_dimensions(ptf_env)
    check_reset(ptf_env)    
    check_done()

    # Test the portfolio calculation, benchmark calculation, and trans. cost calculations
    test_portfolio_value_calculations()
    
    # Test the distributions
    test_distributions()
    
    # Test parsing observations
    test_parsing_observations()
    
    print( 'Tests have run successfully.' )      

def check_space_dimensions(ptf_env):
    # 1 for timestamp, 1 for ptf. val. + 2 for weights 
    #    + 2 for exp. rtns + 3 for lower-diag matrix of risky assets
    assert ptf_env.observation_space.shape == (9,), 'Inconsistent obs. space dims.'
    assert ptf_env.action_space.shape == (3,), 'Inconsistent act, space dims.'
    assert isinstance( ptf_env.observation_space, gym.spaces.Box ), 'Incorrect obs. space type.'
    assert isinstance( ptf_env.action_space, gym.spaces.Box ), 'Incorrect act. space type.'
    
def check_reset(ptf_env):
    ptf_env = portfolio.PortfolioEnv(n_risky_assets=1)
    obs = ptf_env.reset()
    info = ptf_env.parse_observation(obs)
    assert isinstance(info, portfolio.StateVariables), 'Parsing observation does not work.'
    
def check_done():
    N = 3
    ptf_env = portfolio.PortfolioEnv(n_risky_assets=N)
    obs = ptf_env.reset()
    max_timesteps = ptf_env.n_years * ptf_env.n_periods_per_year
    for j in range(max_timesteps):
        rand_action = np.random.uniform(0, 1, N)
        obs, reward, done, info = ptf_env.step(rand_action)
        
    assert isinstance( obs, np.ndarray ), 'Incorrect output type for observation.'
    assert isinstance( reward, np.float ), 'Incorrect output type for info.'    
    assert isinstance( info, dict ), 'Incorrect output type for info.'
    assert done, 'Test should have concluded on the final run.'
    
    parsed = ptf_env.parse_observation(obs)
    assert parsed.timestamp == max_timesteps, 'Timestamp has incorrect value.'
    
    
def check_portfolio_val_calculation(n_risky_assets, trans_costs=0.0, benchmark_weights=None):
    # Initialize the environment and get the first observation
    ptf_env = portfolio.PortfolioEnv( n_risky_assets=n_risky_assets, 
                                      trans_costs=trans_costs, 
                                      benchmark_weights=benchmark_weights, 
                                      min_short=0.0, 
                                      max_long=1.0 )
    obs_0 = ptf_env.parse_observation( ptf_env.reset() )
    ptf_asset_vals_0 = ptf_env.state_vars.ptf_asset_vals
    bmk_asset_vals_0 = ptf_env.state_vars.bmk_asset_vals

    # Generate a random action (weight change)
    action = np.random.uniform(0, 1, n_risky_assets)
    obs_1, _, _, info = ptf_env.step(action)

    asset_rtns = info['asset_rtns']
    wt_chg = info[ 'chg_wts' ]

    # Pay for transaction costs out of the cash allocation
    tot_ptf_val_0 = ptf_asset_vals_0.sum()
    tot_trans_cost = tot_ptf_val_0 * ( wt_chg[1:] * ptf_env.trans_costs ).sum()
    asset_asset_vals_ex_cost_0 = ptf_asset_vals_0 + (wt_chg * tot_ptf_val_0)
    asset_asset_vals_ex_cost_0[0] -= tot_trans_cost
    ptf_asset_vals_1 = asset_asset_vals_ex_cost_0.ravel() * (1 + asset_rtns.ravel() )
    assert np.all( np.isclose( ptf_asset_vals_1, ptf_env.state_vars.ptf_asset_vals ) ), \
                              'Portfolio asset values disagree.'

    if ptf_env.benchmark_weights is not None:
        bmk_asset_vals_1 = bmk_asset_vals_0 * (1 + asset_rtns.ravel() )
        assert np.all( np.isclose( bmk_asset_vals_1, ptf_env.state_vars.bmk_asset_vals ) ), \
                              'Portfolio asset values disagree.'
                  
def test_portfolio_value_calculations():
    check_portfolio_val_calculation( n_risky_assets=1, trans_costs=0.00, benchmark_weights=None)
    check_portfolio_val_calculation( n_risky_assets=10,trans_costs=0.00, benchmark_weights=None)
    check_portfolio_val_calculation( n_risky_assets=1, trans_costs=0.10, benchmark_weights=None)
    check_portfolio_val_calculation( n_risky_assets=4, trans_costs=0.01, benchmark_weights=None)
    check_portfolio_val_calculation( n_risky_assets=1, trans_costs=0.0, \
                                    benchmark_weights=np.array( [0.7, 0.3] ) )
    check_portfolio_val_calculation( n_risky_assets=5, trans_costs=0.0, \
                                    benchmark_weights=1/6 * np.ones((6,)) )
    check_portfolio_val_calculation( n_risky_assets=1, trans_costs=0.04, \
                                    benchmark_weights=np.array( [0.3, 0.7] ) )
    check_portfolio_val_calculation( n_risky_assets=8, trans_costs=0.20, \
                                    benchmark_weights=1/9 * np.ones((9,)) )      
    
def test_distributions():
    test_distribution_normal()
    test_distribution_lognormal()    
    
def test_distribution_normal():
    # Test 1-d distribution
    ran_gen = np.random.RandomState(42)
    distrib = distributions.NormalDistribution(ran_gen, n_periods_per_year=12, \
                                               mu=[0], sigma=[[1]] )
    R = distrib.random(1000)
    assert np.isclose( R.mean(), 0.005580683816504251 ), 'Random normal mean mismatch.'
    assert np.isclose( R.std(), 0.2825339197529383 ), 'Random normal std. dev. mismatch.'

    # Test 2-d distribution
    ran_gen = np.random.RandomState(100)
    distrib = distributions.NormalDistribution(ran_gen, n_periods_per_year=12, \
                                               mu=[0, 2], sigma=[[1, 0.5], [0.5, 1 ]] )
    R = distrib.random(1000)
    assert np.all( np.isclose( R.mean(axis=0), \
                          np.array( [-0.00115286, 0.15694211 ] ) ) ), \
                                        'Random normal means mismatch.'
    assert np.all( np.isclose( np.corrcoef(R.T), \
                    np.array([[1., 0.49051492], [0.49051492, 1.]]) ) ), \
                                        'Random normal correlations mismatch.'   
    
def test_distribution_lognormal():
    # Test 1-d distribution
    ran_gen = np.random.RandomState(128)
    distrib = distributions.LognormalDistribution(ran_gen, n_periods_per_year=12, \
                                                      mu=[0], sigma=[[1]] )
    R = distrib.random(1000)
    assert np.isclose( R.mean(), 0.0569368561232835 ), 'Random normal mean mismatch.'
    assert np.isclose( R.std(), 0.30380553987317443 ), 'Random normal std. dev. mismatch.'

    # Test 2-d distribution
    ran_gen = np.random.RandomState(251)
    distrib = distributions.NormalDistribution(ran_gen, n_periods_per_year=12, \
                                                      mu=[0, 2], sigma=[[1, 0.5], [0.5, 1 ]] )
    R = distrib.random(1000)
    assert np.all( np.isclose( R.mean(axis=0), \
                          np.array( [-0.01548131,  0.15474003] ) ) ), \
                                        'Random normal means mismatch.'
    assert np.all( np.isclose( np.corrcoef(R.T), \
                    np.array([[1., 0.49361069], [0.49361069, 1.]]) ) ), \
                                        'Random normal correlations mismatch.'  
    
def test_parsing_observations():
    # Case with 1 risky asset and without a benchmark
    ptf_env = portfolio.PortfolioEnv(n_risky_assets=1)
    obs = ptf_env.reset()
    assert obs.size == 5
    ps = ptf_env.parse_observation(obs)
    assert ps.timestamp == obs[0], 'Timestamp is not parsed correctly.'
    assert np.all( ps.ptf_asset_vals == obs[1:3] ), 'Portfolio asset values are not parsed correctly.'
    assert ps.bmk_asset_vals is None, 'Benchmark asset values are not parsed correctly.'
    assert ps.process_params['mu'] == obs[3], 'Distr. mean is not parsed correctly.'
    assert ps.process_params['sigma'] == obs[4] ** 2, 'Distr. variance is not parsed correctly.'

    # Case with 1 risky asset and with a benchmark    
    ptf_env = portfolio.PortfolioEnv(n_risky_assets=1, benchmark_weights=[0.4, 0.6])
    obs = ptf_env.reset()
    assert obs.size == 7
    ps = ptf_env.parse_observation(obs)
    assert ps.timestamp == obs[0], 'Timestamp is not parsed correctly.'
    assert np.all( ps.ptf_asset_vals == obs[1:3] ), 'Portfolio asset values are not parsed correctly.'
    assert np.all( ps.bmk_asset_vals == obs[3:5] ), 'Benchmark asset values are not parsed correctly.'
    assert ps.process_params['mu'] == obs[5], 'Distr. mean is not parsed correctly.'
    assert ps.process_params['sigma'] == obs[6] ** 2, 'Distr. variance is not parsed correctly.'

    # Case with 2 risky assets and without a benchmark    
    ptf_env = portfolio.PortfolioEnv(n_risky_assets=2, benchmark_weights=[0.2, 0.6, 0.2])
    obs = ptf_env.reset()
    assert obs.size == 12
    ps = ptf_env.parse_observation(obs)
    assert ps.timestamp == obs[0], 'Timestamp is not parsed correctly.'
    assert np.all( ps.ptf_asset_vals == obs[1:4] ), 'Portfolio asset values are not parsed correctly.'
    assert np.all( ps.bmk_asset_vals == obs[4:7] ), 'Benchmark asset values are not parsed correctly.'
    assert np.all( ps.process_params['mu'] == obs[7:9] ), 'Distr. mean is not parsed correctly.'
    s_11, s_12, s_22 = obs[9:]
    C = np.array( [[s_11, 0], [s_12, s_22] ] )
    S = np.matmul( C, C.T )
    assert np.all( ps.process_params['sigma'] == S ), 'Distr. variance is not parsed correctly.'