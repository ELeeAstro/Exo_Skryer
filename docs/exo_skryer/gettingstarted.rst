***************
Getting started
***************


References
==========



You'll probably use `~exo_skryer.main` eventually, though perhaps it should have a different name.

It's a good idea to experiment nested sampling, so try `~exo_skryer.run_nested_jaxns`

.. code-block:: python

    from exo_skryer import run_nested_jaxns

    run_nested_jaxns()


Plot
====

First plot
----------

Here we make a plot, how wonderful:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt

    plt.plot(range(10), range(10))
    plt.xlabel('Wavenumber')
    plt.ylabel('Flux')
    plt.show()


Second plot
-----------

This one is cool too, but it doesn't show the source code.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    plt.scatter(np.random.uniform(size=1000), np.random.uniform(size=1000))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
