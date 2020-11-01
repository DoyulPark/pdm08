import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

    # Sidebar
    st.sidebar.header("About -")
    st.sidebar.info("pdm08_st_Mid_exam")

def main():
  st.title("Mid_exam by pdm08, 박도율")
  st.header("< Diabetes EDA project >")
  
  # Read Data
  df = pd.read_csv("https://github.com/Redwoods/Py/raw/master/pdm2020/my-note/py-pandas/data/diabetes.csv")
  
  # Show Dataset
  if st.checkbox("Show Dataset"):
    st.dataframe(df.head())
    st.dataframe(df.describe())
  

  # Plot and Visualization
  st.subheader("- Data Visualization")
  
  # #Checkbox
  # # Histograms
  # if st.checkbox("Histograms"):
  #   plt.rcParams['figure.figsize'] = [12, 10];
  #   st.set_option('deprecation.showPyplotGlobalUse', False)
  #   st.write(df.hist())
  #   st.pyplot()

  # # Density plots for all attributes
  # if st.checkbox("Density plots for all attributes"):
  #   st.set_option('deprecation.showPyplotGlobalUse', False)
  #   st.write(df.plot(kind='density', subplots=True, layout=(3,3), sharex=False))
  #   st.pyplot()

  # # Box and Whisker Plots
  # if st.checkbox("Box and Whisker Plots"):
  #   st.set_option('deprecation.showPyplotGlobalUse', False)
  #   st.write(df.plot(kind= 'box', subplots=True, layout=(3,3), sharex=False, sharey=False))
  #   st.pyplot()

  # Selectbox
  type_of_plot = st.selectbox("Select Type of Plot",["Histograms","Density plots for all attributes","Box and Whisker Plots"])
  if st.button("Generate Plot"):
    st.success("You choose {}!".format(type_of_plot))

    if type_of_plot == 'Histograms':
      plt.rcParams['figure.figsize'] = [12, 10]
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.write(df.hist())
      st.pyplot()

    elif type_of_plot == 'Density plots for all attributes':
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.write(df.plot(kind='density', subplots=True, layout=(3,3), sharex=False))
      st.pyplot()

    elif type_of_plot == 'Box and Whisker Plots':
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.write(df.plot(kind= 'box', subplots=True, layout=(3,3), sharex=False, sharey=False))
      st.pyplot()

  # Customizable Plot
  st.subheader("- Cutomizable Plot")
  all_columns_names = df.columns.tolist()
  type_of_cplot = st.selectbox("Select Type of Plot",["bar","line","hist","box"])
  selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

  if st.button("Generate Customized Plot"):
    st.success("Generating Customizable Plot of {} for {}".format(type_of_cplot,selected_columns_names))

    if type_of_cplot == 'bar':
      cust_data = df[selected_columns_names]
      st.bar_chart(cust_data)

    elif type_of_cplot == 'line':
      cust_data = df[selected_columns_names]
      st.line_chart(cust_data)

    elif type_of_cplot:
      cust_plot = df[selected_columns_names].plot(kind=type_of_cplot)
      st.set_option('deprecation.showPyplotGlobalUse', False)
      st.write(cust_plot)
      st.pyplot()


  
  st.write('***')
  
  if st.button("Thanks!"):
    st.balloons()

    st.markdown("* * *")

if __name__ == '__main__':

