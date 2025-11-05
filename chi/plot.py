import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches


def opt(t, x1, x2,x3,  x_f, epsilon, u_n, u_norms, uc_norms, state_folder, norm_folder):

    t_axis = range(t+1)
    plt.figure(figsize = (10,6))

    plt.plot(x1[0, :], x1[1, :], 'b-o', label = 'state')
    plt.plot(x2[0, :], x2[1,:], 'g-o', label = 'state in excat cons')
 

    plt.scatter(x_f[0], x_f[1], color='red', marker='*', s=150, label='x_f (desired)')
    plt.scatter(x1[0, 0], x1[1, 0], color='green', marker='x', s=100, label='Start')
    plt.scatter(x1[0,-1], x1[1, -1], color = 'red', marker = 'x', s= 150, label = 'final')
    plt.scatter(x2[0, -1], x2[1, -1], color = 'red', marker='x', s = 150, label = 'final2')
    plt.scatter(x3[0, -1], x3[1, -1], color = 'blue', marker = 'x', s = 150, label = 'final_cheb')
    circle = patches.Circle((x_f[0], x_f[1]), epsilon, 
                        fill=False, linestyle='--', color='k', label=f'ε-ball (ε={epsilon})')
    plt.gca().add_patch(circle)


    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid(True)
    plt.title('state deviation = {}, epsilon = {}'.format(np.linalg.norm(x_f - x1[-1, -1]), epsilon))
    plt.legend()
    filename = os.path.join(state_folder, 'state_{}.png'.format(t))
    plt.savefig(filename)
    plt.close()


    # Plot input norm on the same figure (secondary y-axis)
    plt.figure(figsize = (10, 6))

    plt.plot(range(1, t+1), u_n, 'k--s', label='‖uₜ‖₂')
    plt.plot(range(1, t+1), u_norms, marker='o', linewidth=2, label ='exact constraint')
    plt.plot(range(1, t+1), uc_norms, marker ='x', color = 'red', label='u_cheb')

    plt.xlabel('time step')
    plt.ylabel('u-norm')
    # plt.title('u norm = {}, u2_norm = {}:'.format(np.linalg.norm(u), np.linalg.norm(u_exact)**2))

    plt.grid(True)
    plt.legend()
    filename1 = os.path.join(norm_folder, 'norm_{}.png'.format(t))
    plt.savefig(filename1)
    plt.close()

# -------------------------------------------------------------
    # Plot results

def diff(T_list, u_norm_diff_list, state_dev_diff_list, diff_folder):
    plt.figure(figsize=(8,5))
    plt.plot(T_list, u_norm_diff_list, 'o-', label='Δ‖u‖ (exact - convex)')
    plt.xlabel('Time horizon T')
    plt.ylabel('Difference in total ‖u‖')
    plt.title('Control norm difference vs Time horizon')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename2 = os.path.join(diff_folder, 'u_norm_diff_vs_T.png')
    plt.savefig(filename2)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(T_list, state_dev_diff_list, 's-', color='purple', label='‖x_T_exact - x_T_approx‖₂')
    plt.xlabel('Time horizon T')
    plt.ylabel('Final state deviation difference')
    plt.title('State deviation difference vs Time horizon')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename3 = os.path.join(diff_folder, 'state_dev_diff_vs_T.png' )
    plt.savefig(filename3)
    plt.close()




