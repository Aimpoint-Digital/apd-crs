.. Borrowed style from existing sci-kit survival package


|apd| apd-crs
============================
.. |apd| image:: apd.jpg
   :width: 100px
   :height: 100px
   :scale: 100 %
   

apd-crs is a Python package for survival analysis with cure rate. I.e. time-to-event
analysis using a dataset for which outcomes of a fraction of the
dataset are unknown (censored), and for which some individuals never
experience the event. For example, a medical trial where time since diagnosis is
measured and some participants drop out during the trial period, some of which are
cured. Or a manufacturing dataset where time-to-failure since maintenance is measured,
and some equipment takes so long to fail that it "never sees" the event. Currently there are three 
methods available for estimating the cure probability per covariate: either by k-means clustering, 
using the `selected completely at random (SCAR) <https://dl.acm.org/doi/pdf/10.1145/1401890.1401920/>`_ assumption, 
or using a `Hard EM algorithm <https://link.springer.com/article/10.1007/s00180-021-01140-0/>`_. The overall survivor
function is fitted by assuming that the lifetime of a susceptible (non-cured) individual follows a
proportional hazards model, and the baseline hazard function is given via Weibull distribution.
Since the cure probability is fitted independently of the times to event/censoring, this package can also be used 
for `PU learning <https://en.wikipedia.org/wiki/One-class_classification#PU_learning>`_. I.e. binary classification with positive and unlabeled data.  
apd-crs is a work in progress and maintained by a team of researchers and
developers from `Aimpoint Digital <https://aimpointdigital.com/>`_.




.. raw:: html

    <div class="row">
      
        <div class="flex-fill tile">
          <!a class="tile-link" href="install.html">
            <h3 class="tile-title">Install
            <i class="fas fa-download tile-icon"></i>
            </h3>
          </a>
          <div class="tile-desc">


Using pip by running::

  pip install -i https://test.pypi.org/simple/ apd-crs




.. raw:: html

          </div>
        </div>
      </div>
      <div class="row">
        <div class="flex-fill tile">
          <a class="tile-link" href="tutorials/index.html">
            <h3 class="tile-title">Step-by-step Tutorials
            <i class="fas fa-book-open tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>Step-by-step tutorials grounded in real world examples from manufacturing, churn, and marketing.</p>
            </div>
          </a>
        </div>
      </div>
      <div class="row">
        <div class="flex-fill tile">
          <a class="tile-link" href="api/index.html">
            <h3 class="tile-title">API reference
            <i class="fas fa-cogs tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>The API reference documents the user exposed methods and parameters.</p>
            </div>
          </a>
        </div>
      </div>
      <div class="row">
        <div class="flex-fill tile">
          <!a class="tile-link" href="contributing.html">
            <h3 class="tile-title">Contributing
            <i class="fas fa-code tile-icon"></i>
            </h3>
            <div class="tile-desc">
              <p>Found typos? Interested in new functionalities? We are working on it so stay tuned!</p>
            </div>
          </a>
        </div>
      </div>
    </div>


.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   api/index
   tutorials/index
   




