from runners import ScenarioRunner

def main(args):

    # towns = {i: f'Town{i+1:02d}' for i in range(7)}
    # towns.update({7: 'Town10HD'})
    towns={0:'Town01', 1 : 'Town03', 2: 'Town04', 3: 'Town06'}

    # print(f'towns: {towns}')
    scenario = 'assets/training_towns_traffic_scenarios.json'
    # scenario = 'assets/no_scenarios.json'
    route = 'assets/routes_training.xml'
    # route = 'assets/routes_training/route_10.xml'

    args.agent = 'autoagents/collector_agents/q_collector' # Use 'viz_collector' for collecting pretty images
    args.agent_config = 'config.yaml'
    # args.agent_config = 'experiments/config_nocrash.yaml'

    # args.agent = 'autoagents/collector_agents/lidar_q_collector'
    # args.agent_config = 'config_lidar.yaml'

    jobs = []
    j = 0
    for i in range(args.num_runners):
        # scenario_class = 'train_scenario' # Use 'nocrash_train_scenario' to collect NoCrash training trajs
        scenario_class = args.scenario
        # print(f'scenarios_calss: {scenario_class}')
        town = towns.get(i) #, 'Town03')
        # print(f'town: {town}')
        port = 2*j + i + args.port
        tm_port = port + 2
        checkpoint = f'results/{i:02d}_{args.checkpoint}'
        # print(f'checkpoint: {checkpoint}')
        runner = ScenarioRunner.remote(args, scenario_class, scenario, route, checkpoint=checkpoint, town=town, port=port, tm_port=tm_port)
        jobs.append(runner.run.remote())
        j = j + 1
    
    ray.wait(jobs, num_returns=args.num_runners)


if __name__ == '__main__':

    import argparse
    import ray
    ray.init(logging_level=40, local_mode=False, log_to_driver=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-runners', type=int, default=8)
    parser.add_argument('--scenario', choices=['train_scenario', 'nocrash_train_scenario'], default='train_scenario')
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="600.0",
                        help='Set the CARLA client timeout value in seconds')

    # agent-related options
    # parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    # parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")
    parser.add_argument('--repetitions',
                        type=int,
                        default=100,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='MAP', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='data_phase1_simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    
    args = parser.parse_args()
    
    main(args)
