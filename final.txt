import streamlit as st
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import re
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import poisson
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import plotly as py
import plotly.graph_objs as go
from numpy import arange,array,ones
from scipy import stats
from scipy.stats import t
from scipy.stats import uniform

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache

def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"

def normalProbabilityDensity(x):
    constant = 1.0 / np.sqrt(2*np.pi)
    return(constant * np.exp((-x**2)/2.0))


st.sidebar.header('User Input Features')
st.title('Final Project Python')
st.write("""
### <Kapizov Dastan --- Dair Maxat --- Kassymbayev Kuanysh>

""")

cumulative_from_mean = st.sidebar.checkbox("Cumulative from mean z-table")
cumulative = st.sidebar.checkbox("Cumulative z-table")
complementary_cumulative = st.sidebar.checkbox("Complementary cumulative z-table")


if cumulative_from_mean:
#cumulative from mean
    st.write("""
    # Cumulative from mean z-table
    """)
    standard_normal_table3 = pd.DataFrame(data = [], index = np.round(np.arange(0, 4.1,.1),2),
                                        columns = np.round(np.arange(0.00, .1, .01),2))
    for index in standard_normal_table3.index:
        for column in standard_normal_table3.columns:
            z = np.round(index + column,2)
            value, _ = quad(normalProbabilityDensity, np.NINF, z)
            standard_normal_table3.loc[index, column] = toFixed(value-0.5,5)

    standard_normal_table3.index = standard_normal_table3.index.astype(str)
    standard_normal_table3.columns = [str(column).ljust(4,'0') for column in standard_normal_table3.columns]


    standard_normal_table3

if cumulative:
##cumulative
    st.write("""
    # Cumulative z-table
    """)
    standard_normal_table = pd.DataFrame(data = [], index = np.round(np.arange(-4,0.1,.1),2),
                                       columns = np.round(np.arange(-0.00, -.1, -.01),2))
    for index in standard_normal_table.index:
        for column in standard_normal_table.columns:
           z = np.round(index + column,2)
           value, _ = quad(normalProbabilityDensity, np.NINF, z)
           standard_normal_table.loc[index, column] = toFixed(value,5)

    standard_normal_table.index = standard_normal_table.index.astype(str)
    standard_normal_table.columns = [str(column).ljust(4,'0') for column in standard_normal_table.columns]

    standard_normal_table

if complementary_cumulative:
#complementary cumulative
    st.write("""
    # Complementary cumulative z-table
    """)
    standard_normal_table2 = pd.DataFrame(data = [], index = np.round(np.arange(0, 4.1,.1),2),
                                        columns = np.round(np.arange(0.00, .1, .01),2))
    for index in standard_normal_table2.index:
        for column in standard_normal_table2.columns:
            z = np.round(index + column,2)
            value, _ = quad(normalProbabilityDensity, np.NINF, z)
            standard_normal_table2.loc[index, column] = toFixed(1-value,5)

    standard_normal_table2.index = standard_normal_table2.index.astype(str)
    standard_normal_table2.columns = [str(column).ljust(4,'0') for column in standard_normal_table2.columns]


    standard_normal_table2



lst = ['<Select>','Normal','Poisson','Gamma','Exponential','Uniform']

distribution = st.sidebar.selectbox('Distribution',lst)

if distribution == 'Normal' :

    mean_input = st.sidebar.number_input('Write the mean')

    scale_input = st.sidebar.number_input('Write the standard deviation')

    X1 = st.sidebar.number_input('X1')

    X2 = st.sidebar.number_input('X2')


    st.write("""
    # Normal Distribution
    """)
    st.write('Normal Distribution, also known as Gaussian distribution, is ubiquitous in Data Science. You will encounter it at many places especially in topics of statistical inference. It is one of the assumptions of many data science algorithms too.')

    st.markdown(r"""
    $$f(z) = \frac{1}{2\pi}\exp(\frac{-z^2}{2})$$
    """)
    st.write("""
    The parameter $\mu$  is the mean or expectation of the distribution (and also its median and mode), while the parameter $\sigma$  is its standard deviation. The variance of the distribution is $\sigma{^2}$. A random variable with a Gaussian distribution is said to be normally distributed, and is called a normal deviate.
    """)
    if st.sidebar.button('Show me'):
        if X2 == 0 and X1 != 0:
            less=norm.cdf(x=X1, loc=mean_input, scale=scale_input)
            fig, ax = plt.subplots()
            x = np.linspace(mean_input - 3*scale_input, mean_input + 3*scale_input, 100)
            ax.plot(x, norm.pdf(x, loc = mean_input, scale = scale_input))
            ax.set_title(f"Normal Dist. with mean = {mean_input}, std_dv = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')

            # for fill_between
            px=np.arange(mean_input - 3*scale_input,X1, 0.01)
            ax.fill_between(px, norm.pdf(px, loc = mean_input, scale = scale_input), alpha = 0.5, color = 'r')

            # for text
            ax.text(X1 - 2, 0.035, f"P(X<{X1})\n%.2f"%round(less, 3), fontsize = 20)
            st.pyplot()
        elif X1 == 0 and X2 != 0:
            more=norm.sf(x=X2, loc=mean_input, scale=scale_input)
            fig, ax = plt.subplots()
            x = np.linspace(mean_input - 3*scale_input, mean_input + 3*scale_input, 100)
            ax.plot(x, norm.pdf(x, loc = mean_input, scale = scale_input))
            ax.set_title(f"Normal Dist. with mean = {mean_input}, std_dv = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')

            # for fill_between
            px=np.arange(X2,mean_input + 3*scale_input, 0.01)

            ax.fill_between(px, norm.pdf(px, loc = mean_input, scale = scale_input), alpha = 0.5, color = 'm')

            # for text
            ax.text(X2 + 0.5, 0.02, f"P(X>{X2})\n%.2f" % round(more, 3), fontsize = 20)
            st.pyplot()
        elif X1 != 0 and X2 != 0:
            between = norm(mean_input, scale_input).cdf(X1) - norm(mean_input, scale_input).cdf(X2)
            fig, ax = plt.subplots()
            x = np.linspace(mean_input - 3*scale_input, mean_input + 3*scale_input, 100)
            ax.plot(x, norm.pdf(x, loc = mean_input, scale = scale_input))
            ax.set_title(f"Normal Dist. with mean = {mean_input}, std_dv = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')

            # for fill_between
            px=np.arange(X2,X1, 0.01)
            ax.fill_between(px, norm.pdf(px, loc = mean_input, scale = scale_input), alpha = 0.5, color = 'c')

            # for text
            ax.text(X2 +0.5, 0.075, f"P({X2}<X<{X1})\n%.2f" % round(between, 3), fontsize = 20)
            st.pyplot()
if distribution == 'Poisson' :

    mean_input = st.sidebar.number_input('Average rate of success')

    X1 = st.sidebar.number_input("P(X <= x)")

    X2 = st.sidebar.number_input('P(X > x)')


    st.header('Poisson Distribution')

    st.write("""
    Poisson random variable is typically used to model the number of times an event happened in a time interval. For example, the number of users visited on a website in an interval can be thought of a Poisson process. Poisson distribution is described in terms of the rate (μ) at which the events happen. An event can occur 0, 1, 2, … times in an interval. The average number of events in an interval is designated λ (lambda). Lambda is the event rate, also called the rate parameter. The probability of observing k events in an interval is given by the equation:
    """)
    st.markdown(r"""
    $$P(k) = e^{-\lambda}\frac{\lambda^k}{k!}$$   where k is number of events.
    """)
    st.write("""

    Note that the normal distribution is a limiting case of Poisson distribution with the parameter $\lambda \ -> \infty$. Also, if the times between random events follow an exponential distribution with rate $\lambda$, then the total number of events in a time period of length t follows the Poisson distribution with parameter $\lambda t$.
    """)

    if st.sidebar.button('Show me'):
        if X2 == 0 and X1 > 0:

            less = poisson.cdf(X1, mu=mean_input,loc = 0)
            fig, ax = plt.subplots()

            sns.distplot(np.random.poisson(lam=mean_input, size=1000), hist=False, label='poisson')
            ax.set_title(f"Poisson Dist. with mean = {mean_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(0,X1+1, 1)

            ax.fill_between(px, poisson.pmf(px, mu = mean_input, loc = 0), alpha = 0.5, color = 'r')

            # for text
            ax.text(X1, 0.075, f"P(X<={X1})\n%.2f" %round(less, 3), fontsize = 20)
            st.pyplot()
        elif X1 == 0 and X2 > 0:
            more = poisson.sf(X2, mu = mean_input)
            fig, ax = plt.subplots()
            sns.distplot(np.random.poisson(lam=mean_input, size=1000), hist=False, label='poisson')

            ax.set_title(f"Poisson Dist. with mean = {mean_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(X2,14, 1)
            ax.fill_between(px, poisson.pmf(px, mu = mean_input, loc = 0), alpha = 0.5, color = 'm')

            # for text
            ax.text(X2+2, 0.02, f"P(X>{X2})\n%.2f" %round(more, 3), fontsize = 20)
            st.pyplot()
        elif X1 > 0 and X2 > 0:
            between = poisson.cdf(X1, mu = mean_input) - poisson.cdf(X2-1, mu = mean_input)
            fig, ax = plt.subplots()
            sns.distplot(np.random.poisson(lam=mean_input, size=1000), hist=False, label='poisson')

            ax.set_title(f"Poisson Dist. with mean = {mean_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')

            # for fill_between
            px=np.arange(X2,X1+1, 1)
            ax.fill_between(px, poisson.pmf(px, mu = mean_input, loc = 0), alpha = 0.5, color = 'c')

            # for text
            ax.text(X1 - 2, 0.075,f"P({X2}<=X<={X1})\n%.2f" % round(between, 3), fontsize = 20)
            st.pyplot()

if  distribution == 'Gamma' :
    mean_input = st.sidebar.number_input('Write the mean')

    scale_input = st.sidebar.number_input('Write the standard deviation')

    X1 = st.sidebar.number_input('X1')

    X2 = st.sidebar.number_input('X2')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("""
    # Gamma Distribution
    """)


    st.write('The gamma distribution is a two-parameter family of continuous probability distributions. While it is used rarely in its raw form but other popularly used distributions like exponential, chi-squared, erlang distributions are special cases of the gamma distribution.')
    st.markdown(r"""
    $$f(x_{i})=\frac{1}{scale^{shape}\Gamma(shape)}x_{i}^{(shape-1)}e^{-(\frac{x_{i}}{scale})}$$    """)
    st.markdown(r"""
    The Gamma
    function arises in many statistical applications. The formula appears
    to be complicated, but just remember: its just the factorial function
    ``extended'' to take on values between the integers.
    """)
    if st.sidebar.button('Show me'):
        if X2 == 0 and X1 > 0:
            less=gamma.cdf(x=X1, a=mean_input, scale=scale_input)
            fig, ax = plt.subplots()
            x = np.arange(0, 12, 0.001)
            ax.plot(x, gamma.pdf(x, a = mean_input, scale = scale_input))
            ax.set_title(f"Gamma Dist. with mean = {mean_input}, std_dv = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(0,X1, 0.01)

            ax.fill_between(px, gamma.pdf(px, a = mean_input, scale = scale_input), alpha = 0.5, color = 'r')

            # for text
            ax.text(1.0, 0.075, f"P(X<{X1})\n%.2f"%round(less, 2), fontsize = 20)
            st.pyplot()
        elif X1 == 0 and X2 > 0:
            more=gamma.sf(x=X2, a=mean_input, scale=scale_input)
            fig, ax = plt.subplots()
            x = np.arange(0, 12, 0.001)
            ax.plot(x, gamma.pdf(x, a = mean_input, scale = scale_input))
            ax.set_title(f"Gamma Dist. with mean = {mean_input}, std_dv = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(X2,12, 0.01)

            ax.fill_between(px, gamma.pdf(px, a = mean_input, scale = scale_input), alpha = 0.5, color = 'm')

            # for text
            ax.text(4, 0.02, f"P(X>{X2})\n%.2f"%round(more, 2), fontsize = 20)
            st.pyplot()
        elif X1 > 0 and X2 > 0:
            between = gamma(mean_input, scale_input).cdf(X2) - gamma(mean_input, scale_input).cdf(X1)
            fig, ax = plt.subplots()
            x = np.arange(0, 12, 0.001)
            ax.plot(x, gamma.pdf(x, a = mean_input, scale = scale_input))
            ax.set_title(f"Gamma Dist. with mean = {mean_input}, std_dv = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(X1,X2, 0.01)

            ax.fill_between(px, gamma.pdf(px, a = mean_input, scale = scale_input), alpha = 0.5, color = 'c')

            # for text
            ax.text(1.15, 0.075, f"P({X1}<X<{X2})\n%.2f" % round(between, 2), fontsize = 20)
            st.pyplot()

if distribution == 'Exponential' :

    scale_input = st.sidebar.number_input('Write the scale')

    X1 = st.sidebar.number_input('X1')

    X2 = st.sidebar.number_input('X2')

    st.write("""
    # Exponential Distribution
    """)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write('The exponential distribution describes the time between events in a Poisson point process, i.e., a process in which events occur continuously and independently at a constant average rate.')
    st.markdown(r"""
    $$f(x) = \lambda e ^ {- \lambda x} \qquad \qquad x > 0,$$ for $\lambda > 0$.""")
    st.markdown(r"""
    The meaning of the rate parameter depends
    on the application (for example, failure rate for reliability, arrival rate or service rate
    for queueing, recidivism rate in criminal justice).
    The exponential distribution is used in reliability to model the lifetime of an
    object which, in a statistical sense, does not age (for example, a
    fuse or light bulb).  This property is known as the memoryless property.
    The exponential distribution is the only continuous distribution that
    possesses this property.
    """)
    if st.sidebar.button('Show me'):
        if X2 == 0 and X1 > 0:
            less=expon.cdf(x=X1, scale=scale_input)
            fig, ax = plt.subplots()
            x = np.arange(0, 12, 0.001)
            ax.plot(x, expon.pdf(x, scale = scale_input))
            ax.set_title(f"Exponential Dist. with  lambda = {1/scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(0,X1, 0.01)
            ax.set_ylim(0, 0.40)
            ax.fill_between(px, expon.pdf(px, scale = scale_input), alpha = 0.5, color = 'r')

            # for text
            ax.text(1.0, 0.075, f"P(X<{X1})\n%.2f"%round(less, 2), fontsize = 20)
            st.pyplot()
        elif X1 == 0 and X2 > 0:
            more=expon.sf(x=X2,scale=scale_input)
            fig, ax = plt.subplots()
            x = np.arange(0, 12, 0.001)
            ax.plot(x, expon.pdf(x, scale = scale_input))
            ax.set_title(f"Exponential Dist. with  lambda = {1/scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(X2,12, 0.01)
            ax.set_ylim(0, 0.40)
            ax.fill_between(px, expon.pdf(px,scale = scale_input), alpha = 0.5, color = 'm')

            # for text
            ax.text(4, 0.02, f"P(X>{X2})\n%.2f"%round(more, 2), fontsize = 20)
            st.pyplot()
        elif X1 > 0 and X2 > 0:
            between = expon(scale = scale_input).cdf(X2) - expon(scale = scale_input).cdf(X1)
            fig, ax = plt.subplots()
            x = np.arange(0, 12, 0.001)
            ax.plot(x, expon.pdf(x,scale = scale_input))
            ax.set_title(f"Exponential Dist. with  lambda = {1/scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')
            ax.set_ylim(0,0.4)
            # for fill_between
            px=np.arange(X1,X2, 0.01)
            ax.set_ylim(0, 0.40)
            ax.fill_between(px, expon.pdf(px,scale = scale_input), alpha = 0.5, color = 'c')

            # for text
            ax.text(1.15, 0.075, f"P({X1}<X<{X2})\n%.2f" % round(between, 2), fontsize = 20)
            st.pyplot()
if distribution == 'Uniform' :

    mean_input = st.sidebar.number_input('Write the lower limit')

    scale_input = st.sidebar.number_input('Write the upper limit')

    X1 = st.sidebar.number_input('X1')

    X2 = st.sidebar.number_input('X2')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("""
    # Uniform Distribution
    """)
    st.write('The uniform distribution is a type of continuous probability distribution that can take random values on the the interval [a, b][a,b], and it zero outside of this interval.')
    st.markdown(r"""
    $$\displaystyle{f{{({x})}}}=\frac{{1}}{{{b}-{a}}}\\$$
    """)
    st.write("""
    The uniform distribution is a continuous probability distribution and is concerned with events that are equally likely to occur. When working out problems that have a uniform distribution, be careful to note if the data is inclusive or exclusive.
    """)

    if st.sidebar.button('Show me'):
        if X2 == 0 and X1 > 0:
            less=uniform.cdf(x=X1, loc=mean_input, scale=scale_input-mean_input)
            fig, ax = plt.subplots()
            x = np.arange(-4, 10, 0.001)
            ax.plot(x, uniform.pdf(x, loc = mean_input, scale = scale_input-mean_input))
            ax.set_title(f"Uniform Dist. with lower = {mean_input}, upper = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')

            # for fill_between
            px=np.arange(-4,X1, 0.01)
            ax.set_ylim(0, 0.40)
            ax.fill_between(px, uniform.pdf(px, loc = mean_input, scale = scale_input-mean_input), alpha = 0.5, color = 'r')

            # for text
            ax.text(1.0, 0.075, f"P(X<{X1})\n%.2f"%round(less, 2), fontsize = 20)
            st.pyplot()
        elif X1 == 0 and X2 > 0:
            more=uniform.sf(x=X2, loc=mean_input, scale=scale_input-mean_input)
            fig, ax = plt.subplots()
            x = np.arange(-4, 10, 0.001)
            ax.plot(x, uniform.pdf(x, loc = mean_input, scale = scale_input-mean_input))
            ax.set_title(f"Uniform Dist. with lower = {mean_input}, upper = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')

            # for fill_between
            px=np.arange(X2,10, 0.01)
            ax.set_ylim(0, 0.40)
            ax.fill_between(px, uniform.pdf(px, loc = mean_input, scale = scale_input-mean_input), alpha = 0.5, color = 'm')

            # for text
            ax.text(4, 0.02, f"P(X>{X2})\n%.2f"%round(more, 2), fontsize = 20)
            st.pyplot()
        elif X1 > 0 and X2 > 0:
            between = uniform(mean_input, scale_input-mean_input).cdf(X2) - uniform(mean_input, scale_input-mean_input).cdf(X1)
            fig, ax = plt.subplots()
            x = np.linspace(mean_input - 3*(scale_input), mean_input + 3*(scale_input), 100)

            ax.plot(x, uniform.pdf(x, loc = mean_input, scale = scale_input-mean_input))
            ax.set_title(f"Uniform Dist. with lower = {mean_input}, upper = {scale_input}")
            ax.set_xlabel('X-Values')
            ax.set_ylabel('PDF(X)')

            # for fill_between
            px=np.arange(X1,X2, 0.01)
            ax.set_ylim(0,0.40 )
            ax.fill_between(px, uniform.pdf(px, loc = mean_input, scale = scale_input-mean_input), alpha = 0.5, color = 'c')

            # for text
            ax.text(X2 +0.5, 0.075, f"P({X1}<X<{X2})\n%.2f" % round(between, 3), fontsize = 20)
            st.pyplot()

if distribution == '<Select>':


    x_input =  st.sidebar.text_input('Please, write x values','50.5,52,53,54.8,58.4,60')
    y_input =  st.sidebar.text_input('Please write y values','25.6,50.7,55,75,80.6,85')
    
    add = st.sidebar.checkbox("Add confidence interval around the regression line",value=True)

    X = re.findall(r"[-+]?\d*\.\d+|\d+", x_input)
    list_of_floats = []
    for num in X:
        list_of_floats.append(float(num))
    X = list_of_floats

    y = re.findall(r"[-+]?\d*\.\d+|\d+", y_input)
    list_of_floats = []
    for num in y:
        list_of_floats.append(float(num))
    y = list_of_floats

    regres_table = pd.DataFrame(data = [], columns = ['x','y'])
    regres_table['x'] = X
    regres_table['y'] = y
    st.table(regres_table)

    import plotly.express as px

   # slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    #line = px.get_trendline_results(s)
    ##rg = sns.regplot(x,y, color ='blue')
    #P=rg.get_children()[1].get_paths()
    ##
    #st.plotly_chart(fig)
    #st."""
    #from plotnine import *
   # from plotnine import ggplot, aes, geom_smooth,geom_point


    #p = ggplot(regres_table, aes(x = x, y =y )) + geom_point() + geom_smooth(color = 'red')
    #st.pyplot(ggplot.draw(p))


    import chart_studio.plotly as py  
    import plotly.graph_objs as go

    from numpy import arange,array,ones
    from scipy import stats

    x = X

    x = np.array(x)


    slope, intercept = np.polyfit(x,y,1)
    line = slope * x + intercept

    rg = sns.regplot(x,y)

    P=rg.get_children()[1].get_paths()


# Creating the dataset, and generating the plot
    
    trace1 = go.Scatter(
                      x=x,
                      y=y,
                      mode='markers',
                      marker=go.Marker(color='rgb(28, 212, 83)'),
                      name='Data'
                      )

    trace2 = go.Scatter(
                      x=x,
                      y=line,
                      mode='lines',
                      marker=go.Marker(color='rgb(31, 119, 180)'),
                      name='Fit'
                    )

    data = [trace1, trace2]
    layout = go.Layout(
                    title='Linear Fit in Python',
                    autosize= False, 
                    hovermode = 'closest', 
                    showlegend= False, 
                    plot_bgcolor='rgb(229, 229, 229)',
                      xaxis=go.XAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)',
                      zeroline =False,
                      mirror = True,
                      ticklen = 4,
                      showline = True 
                      
                      ),
                      yaxis=go.YAxis(zerolinecolor='rgb(255,255,255)', gridcolor='rgb(255,255,255)',
                      zeroline =False,
                      mirror = True,
                      ticklen = 4,
                      showline = True 
                      ),
                    )

    p_codes={1:'M', 2: 'L', 79: 'Z'}#dict to get the Plotly codes for commands to define the svg path
    path=''
    for s in P[0].iter_segments():
        c=p_codes[s[1]]
        xx, yy=s[0]
        path+=c+str('{:.5f}'.format(xx))+' '+str('{:.5f}'.format(yy))

    shapes=[dict(type='path',
                 path=path,
                 line=dict(width=0.1,color='rgba(68, 122, 219, 0.25)' ),
                 fillcolor='rgba(68, 122, 219, 0.25)')]   
    if add:
        layout['shapes']=shapes

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

    from sklearn.model_selection import KFold
    kf=KFold(n_splits=len(X))
    mse_list=[]
    rmse_list=[]
    r2_list=[]
    msetotal = []
    rmsetotal = []
    r2total = []
    Xx = np.array(X).reshape(-1,1)
    yy = np.array(y).reshape(-1,1)
    idx=1
    i=0
    msearr = 0
    rmsearr = 0
    r2arr = 0
    xmean = np.mean(Xx)
    ymean = np.mean(yy)

    for train_index, test_index in kf.split(Xx):
        Xx_train, Xx_test = Xx[train_index],Xx[test_index]
        yy_train, yy_test = yy[train_index],yy[test_index]
        lrgr = LinearRegression()
        lrgr.fit(Xx_train,yy_train)
        pred = lrgr.predict(Xx_test)
        mse = mean_squared_error(yy_test,pred)
        rmse = sqrt(mse)
        r2=r2_score(yy_test,pred)
        RE = (yy - pred)**2
        #Residual Sum Squares
        RSS = RE.sum()
        # Estimated Standard Variation (sigma) or RSE
        RSE = np.sqrt(RSS/(len(X)-2))
        # Total Sum of squares (TSS)
        TE = (yy - ymean)**2
        # Total Sum Squares
        TSS = TE.sum()
        # R^2 Statistic
        R2 = 1 - RSS/TSS
        msearr = msearr + mse
        rmsearr = rmsearr + rmse
        r2arr = r2arr + R2
        toFixed(mse,3)
        toFixed(rmse,3)
        toFixed(R2,3)
        toFixed(msearr,3)
        toFixed(rmsearr,3)
        toFixed(r2arr,3)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(R2)
        plt.plot(pred,label=f"dataset-{idx}")
        idx+=1

    mse_list.append("Total")
    rmse_list.append("Total")
    r2_list.append("Total")
    mse_list.append(msearr/len(X))
    rmse_list.append(rmsearr/len(X))
    r2_list.append(r2arr/len(X))
    msetotal.append(msearr/len(X))
    rmsetotal.append(rmsearr/len(X))
    r2total.append(r2arr/len(X))
    st.header('MSE')
    st.write("""
    In statistics, the mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
    """)
    st.header('RMSE')
    st.write("""
    Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.
    """)
    st.header('R2 score')
    st.write("""
     R2 compares the fit of the chosen model with that of a horizontal straight line (the null hypothesis). If the chosen model fits worse than a horizontal line, then R2 is negative.R2 is negative only when the chosen model does not follow the trend of the data, so fits worse than a horizontal line.
    """)
    score_table = pd.DataFrame(data = [], columns = ['MSE','RMSE','R2'])
    score_table['MSE'] = msetotal
    score_table['RMSE'] = rmsetotal
    score_table['R2'] = r2total
    st.table(score_table)
    if st.sidebar.checkbox('Show by row',False):
        score_table = pd.DataFrame(data = [], columns = ['MSE','RMSE','R2'])
        score_table['MSE'] = mse_list
        score_table['RMSE'] = rmse_list
        score_table['R2'] = r2_list
        st.table(score_table)
    