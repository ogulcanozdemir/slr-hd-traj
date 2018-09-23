from pylab import *

# plot ( arange(0,10),[9,4,5,2,3,5,7,12,2,3],'.-',label='sample1' )
# plot ( arange(0,10),[12,5,33,2,4,5,3,3,22,10],'o-',label='sample2' )
# xlabel('x axis')
# ylabel('y axis')
# title('my sample graphs')
# legend(('sample1','sample2'))
# show()

x = [1, 2, 3, 4, 5]
x_l= [8, 16, 32, 64, 128]
y_traj = [38.26, 45.60, 55.14, 58.28, 62.68]
plot(x, y_traj, '.-', label='Traj', linewidth=.75)

y_hog = [62.58, 69.50, 73.06, 75.89, 79.25]
plot(x, y_hog, '.-', label='HOG', linewidth=.75)

y_hof = [72.75, 76.00, 78.09, 80.08, 81.66]
plot(x, y_hof, '.-', label='HOF', linewidth=.75)

y_mbh = [72.96, 78.51, 81.13, 82.81, 84.91]
plot(x, y_mbh, '.-', label='MBH', linewidth=.75)

y_hog_hof = [79.04, 82.49, 83.96, 85.43, 86.48]
plot(x, y_hog_hof, '.-', label='HOG+HOF', linewidth=.75)

y_hog_mbh = [77.57, 81.34, 83.96, 85.95, 87.53]
plot(x, y_hog_mbh, '.-', label='HOG+MBH', linewidth=.75)

y_hof_mbh = [78.41, 81.03, 82.49, 83.75, 84.49]
plot(x, y_hof_mbh, '.-', label='HOF+MBH', linewidth=.75)

y_hog_hof_mbh = [81.13, 83.75, 85.22, 87.74, 87.84]
plot(x, y_hog_hof_mbh, '.-', label='All w/o Traj', linewidth=.75)

y_traj_hog_hof_mbh = [78.62, 81.45, 82.60, 83.86, 86.27]
plot(x, y_traj_hog_hof_mbh, '.-', label='All', linewidth=.75)

xlabel('Number of clusters (k)')
ylabel('Accuracy (%)')
xticks(x, x_l)
legend(('Traj','HOG', 'HOF', 'MBH', 'HOG+HOF', 'HOG+MBH', 'HOF+MBH', 'All w/o Traj', 'All'), prop={'size': 8})
# show()
savefig("sample.png", dpi=(100))