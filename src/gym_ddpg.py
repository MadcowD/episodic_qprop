import filter_env
from ddpg import *
import gc
gc.enable()

ENV_NAME = 'MountainCarContinuous-v0'
EPISODES = 100000
TEST = 10

def main(summary_dir):
	env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
	agent = DDPG(env)
	#env.monitor.start('experiments/' + ENV_NAME,force=True)

	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(summary_dir + '/train',
										  agent.sess.graph)
	test_writer = tf.summary.FileWriter(summary_dir + '/test',
										  agent.sess.graph)
	train_t = 0
	for episode in range(EPISODES):
		
		state = env.reset()
		#print "episode:",episode
		# Train
		total_reward = 0
		for step in range(env.spec.timestep_limit):
			action = agent.noise_action(state)
			next_state,reward,done,_ = env.step(action)
			total_reward += reward
			agent.perceive(state,action,reward,next_state,done)

			summaries = [
				tf.Summary.Value(tag="q", simple_value=agent.critic_network.q_value([state], [action])[0][0]), 
			]
			summaries += [tf.Summary.Value(tag="reward", simple_value=total_reward)] if done else []
			summary = tf.Summary(value=summaries)
			summaries += [tf.Summary.Value(tag="reward", simple_value=total_reward)]
			
			state = next_state
			if done:
				print("episode: {}, reward: {}".format(episode, total_reward))
				break

		# Testing:
		if episode % 3 == 0 and episode > 10:
			total_reward = 0
			for i in range(TEST):
				state = env.reset()
				for j in range(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			summary = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=ave_reward)])
			test_writer.add_summary(summary, train_t)
			print('episode: ',episode,'Evaluation Average Reward:',ave_reward)
	#env.monitor.close()

if __name__ == '__main__':
	main('/tmp')
