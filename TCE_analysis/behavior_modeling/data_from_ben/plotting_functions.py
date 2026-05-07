import matplotlib.pyplot as plt
from scipy.stats import sem, t

def plot_temp_pain(df, title=None, temp_col='temperature', pain_col='pain', time_col=None):
    """
    Plot temperature (left y-axis) and pain (right y-axis) from a DataFrame.
    
    Parameters:
        df: DataFrame containing the data.
        title: Optional plot title.
        temp_col: Name of the temperature column.
        pain_col: Name of the pain column.
        time_col: Name of the time column (if not index). If None, use df.index.
    """
    if time_col is not None:
        x = df[time_col]
    else:
        x = df.index

    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Temperature on left y-axis
    ax1.plot(x, df[temp_col], color='blue', label='Temperature')
    ax1.set_xlabel("Aligned Time (s)")
    ax1.set_ylabel("Temperature", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Pain on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, df[pain_col], color='red', label='Pain')
    ax2.set_ylabel("Pain", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    if title:
        plt.title(title)
    fig.tight_layout()
    plt.show()


def plot_temp_pain_separate(df, title=None, temp_col='temperature', pain_col='pain', time_col=None):
    """
    Plot temperature and pain separately from a DataFrame.
    
    Parameters:
        df: DataFrame containing the data.
        title: Optional plot title.
        temp_col: Name of the temperature column.
        pain_col: Name of the pain column.
        time_col: Name of the time column (if not index). If None, use df.index.
    """
    if time_col is not None:
        x = df[time_col]
    else:
        x = df.index

    fig, (ax_temp, ax_pain) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Temperature plot
    ax_temp.plot(x, df[temp_col], color='blue', label='Temperature')
    ax_temp.set_ylabel("Temperature", color='blue')
    ax_temp.tick_params(axis='y', labelcolor='blue')
    
    # Pain plot
    ax_pain.plot(x, df[pain_col], color='red', label='Pain')
    ax_pain.set_ylabel("Pain", color='red')
    ax_pain.tick_params(axis='y', labelcolor='red')
    
    if title:
        plt.suptitle(title)
    
    fig.tight_layout()
    plt.show()