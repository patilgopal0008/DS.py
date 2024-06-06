import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np

# Load datasets
datasets = {
    "u shape": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/1.ushape.csv",
    "concentric circle 1": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/2.concerticcir1.csv",
    "concentric circle 2": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/3.concertriccir2.csv",
    "linear sep": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/4.linearsep.csv",
    "outlier": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/5.outlier.csv",
    "overlap": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/6.overlap.csv",
    "xor": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/7.xor.csv",
    "two spirals": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/8.twospirals.csv",
    "random": r"C:\Users\ADMIN\Downloads\ml tasks\Multiple CSV/9.random.csv"
}

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path, header=None)
    return df

def create_ann_model(input_shape, hidden_layers, activations):
    inputs = Input(shape=(input_shape,))
    x = Dense(hidden_layers[0], activation=activations[0])(inputs)
    for units, activation in zip(hidden_layers[1:], activations[1:]):
        x = Dense(units, activation=activation)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrapper class to provide predict method
class ModelPredictWrapper:
    def __init__(self, model):
        self.model = model
    
    def predict(self, x):
        return (self.model.predict(x) > 0.5).astype(int)

# Wrapper class for neuron prediction
class NeuronPredictWrapper:
    def __init__(self, model, layer_idx, neuron_idx):
        self.intermediate_model = Model(inputs=model.input, outputs=model.layers[layer_idx + 1].output[:, neuron_idx])
    
    def predict(self, x):
        return self.intermediate_model.predict(x)

# Streamlit app
st.title("Neural Network Decision Region Visualization")

# Sidebar for dataset selection
with st.sidebar:
    dataset_name = st.selectbox("Select a dataset", list(datasets.keys()))

    # User inputs for the neural network
    with st.form(key='nn_form'):
        num_hidden_layers = st.number_input("Number of hidden layers", min_value=1, max_value=10, value=2)
        hidden_layers = []
        activations = []
        for i in range(num_hidden_layers):
            hidden_layers.append(st.number_input(f"Neurons in hidden layer {i+1}", min_value=1, max_value=100, value=2, key=f"neuron_{i}"))
            activations.append(st.selectbox(f"Activation function for hidden layer {i+1}", ["sigmoid", "tanh", "linear"], index=1, key=f"activation_{i}"))
        
        epochs = st.number_input("Number of epochs", min_value=1, max_value=1000, value=200)
        batch_size = st.number_input("Batch size", min_value=1, max_value=500, value=100)

        submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # Load and prepare the data
    data_path = datasets[dataset_name]
    data = pd.read_csv(data_path)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Standardize the features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    # Create and train the model
    model = create_ann_model(x.shape[1], hidden_layers, activations)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    # Ensure the model is called at least once
    model(x_train)

    # Wrap the main model
    main_model_wrapper = ModelPredictWrapper(model)

    # Plot scatter plot of the data before splitting
    st.subheader("Scatter Plot of the Selected Dataset")
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    st.pyplot(fig)

    # Plot decision regions for the output layer
    st.subheader("Decision Region for Output Layer")
    fig, ax = plt.subplots()
    plot_decision_regions(x_train, y_train.astype(int), clf=main_model_wrapper, legend=2, ax=ax)
    st.pyplot(fig)

    num_cols = max(hidden_layers)
    num_rows = len(hidden_layers)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))

    for layer_idx in range(num_hidden_layers):
        for neuron_idx in range(hidden_layers[layer_idx]):
            if num_hidden_layers > 1:
                ax = axs[layer_idx, neuron_idx]
            else:
                ax = axs[neuron_idx]

            neuron_wrapper = NeuronPredictWrapper(model, layer_idx, neuron_idx)
            ax.set_title(f'Layer {layer_idx + 1} Neuron {neuron_idx + 1}')
            plot_decision_regions(x, y.astype(int), clf=neuron_wrapper, legend=2, ax=ax)
    
    plt.tight_layout()
    st.pyplot(fig)

    # Plot training and validation loss
    st.subheader("Training and Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs+1), history.history["loss"], label="Training Loss")
    ax.plot(range(1, epochs+1), history.history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)