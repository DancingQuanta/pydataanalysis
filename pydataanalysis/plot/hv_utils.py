def layout_subplot_opts(plot, **subplot_kwargs):
    # Per plot opts
    layout_opts = dict(
        show_legend=False,
        ylabel='',
        yaxis='bare')

    count = 0
    for k, v in plot.data.items():

        # Copy kwargs for each plot
        kwargs = subplot_kwargs.copy()

        # Apply title
        if 'title' in subplot_kwargs:
            kwargs['title'] = kwargs['title'].format(k[0])

        # Apply opts to second and rest plots
        if count > 0:
            kwargs.update(layout_opts)

        # update opts
        v.opts(**kwargs)

	# Increment
        count =+ 1

    plot.data[k] = v
    return plot
