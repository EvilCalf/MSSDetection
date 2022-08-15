from nets.MSSDet import MSSDet_model
from keras.utils.vis_utils import plot_model
  

if __name__ == "__main__":
    input_shape = [224, 224, 3]
    num_classes = 2

    model = MSSDet_model(input_shape, num_classes)
    model.summary()
    plot_model(model,to_file='model.png',show_shapes=True)
