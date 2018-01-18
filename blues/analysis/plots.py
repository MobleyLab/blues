def animatePoseSampling(distances,cluster_labels,title='Simulation', outfname='scat_rmsd',fps=30, interval=50, size=50, n_clusters=4, xlim=0, ylim=0, acc_it=[],cmap='gist_rainbow'):
    N = len(cluster_labels)
    fig = plt.figure(3,figsize=(8, 6), dpi=300, tight_layout=True)
    ax = fig.add_subplot(111)
    #plt.figure(figsize=(8, 6), dpi=300,tight_layout=True)
    #f, ax = plt.subplots(111)
    time = [0.0042*t for t in range(N)]
    x1 = np.asarray(time)
    y1 = np.asarray(10.0*distances)
    x2 = time
    y2 = list(10.0*distances)

    if not xlim: xlim = np.max(x1)
    if not ylim: ylim = np.max(y1)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"RMSD $\AA$")
    ax.set_xlim([0,xlim])
    ax.set_ylim([0,ylim])
    if acc_it:
        for it in acc_it:
            ax.axvline(x=it, color='k', linestyle='--')

    NUM_COLORS = n_clusters
    cm = plt.get_cmap(cmap)
    colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

    colr_list = []
    #cmap = get_cmap(n_clusters*2)
    for x in cluster_labels:
        colr_list.append(colors[x])

    plots = [ax.scatter([], [],s=size, c=colr_list),
             ax.plot([], [], 'k-', linewidth=0.25,animated=True)[0]]
    if acc_it:
        for it in acc_it:
            ax.axvline(x=it, color='k', linestyle='--', linewidth=0.5)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='NLM'), bitrate=-1)

    def init():
        for idx,plot in enumerate(plots):
            if idx == 0:
                plot.set_offsets([])

            elif idx ==1:
                plot.set_data([],[])
        return plots

    def animate(i):
        data = np.hstack((x1[:i,np.newaxis], y1[:i, np.newaxis]))
        plots[0].set_offsets(data)
        plots[1].set_data(x2[:i],y2[:i])
        fig.suptitle('\n%s Time = %0.1fns' %(title,time[i]))
        drawProgressBar(i/(N-1))
        return plots

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x1),
                                   interval=50, blit=False, repeat=False)
    anim.save(outfname+'.mp4',writer=writer)
    HTML(anim.to_html5_video())
