library(tensorflow)
library(keras)
library(tfdatasets)
library(caret)
library(abind)
art <- reticulate::import("art")
np <- reticulate::import("numpy")
source('/Users/willi3by/Documents/R/niiMLr/R/custom_residual_layer.R')
source('/Users/willi3by/Documents/R/niiMLr/R/build_ResNet.R')
setwd('/Users/willi3by/Desktop/Parkinsons_SPECT_Project/')

dataset <- tfrecord_dataset('./data/test.tfrecord') %>%
  dataset_map(function(example_proto){
    features <- tf$io$parse_single_example(
      example_proto,
      features <- list(
        'train/image' = tf$io$FixedLenFeature(shape(91,109,91,1), tf$float32),
        'train/label' = tf$io$FixedLenFeature(shape(), tf$int64)
      )
    )
    ##CAN ADD PREPROCESSING HERE.
    list(features$`train/image`, features$`train/label`)
  }) %>%
  dataset_batch(125)

dataset_iter <- tf$compat$v1$data$make_one_shot_iterator(dataset)
nx <- dataset_iter$get_next()

x_test <- nx[[1]]$numpy()
y_test <- nx[[2]]$numpy()

tf$compat$v1$disable_eager_execution()
input_shape <- c(91,109,91,1)
source_model <- build_ResNet(input_shape = c(91,109,91,1), 
                             num_classes = 2, optimizer = optimizer_adam(lr=0.0001), 
                             metrics = list("accuracy"))
source_model <- source_model %>% load_model_weights_hdf5('./model/best_model.h5')
model <- keras_model_sequential() %>%
  layer_conv_3d(filters = 64, kernel_size = c(7,7,7), strides = c(2,2,2), padding = 'same',
                use_bias = F, input_shape = input_shape) %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  layer_max_pooling_3d(pool_size = c(3,3,3), strides = c(2,2,2), padding = 'same')

prev_filters <<- 64
filter_list <<- c(rep(64,3), rep(128, 4), rep(256,6), rep(512,3))
for(i in seq_along(filter_list)){
  i <<- i
  if(filter_list[i] == prev_filters){strides <<- 1} else {strides <<- 2}
  model %>% residual_layer(filters = filter_list[i], strides = strides)
  prev_filters <<- filter_list[i]
}

model <- model %>%
  layer_global_average_pooling_3d() %>%
  layer_flatten() %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 2, activation = "softmax")

model <- model %>% compile(loss="binary_crossentropy",
                           optimizer = optimizer_adam(0.0001),
                           metrics = "accuracy")

for(w in 1:(length(model$layers)-1)){
  source_weights <- source_model$layers[[w]]$get_weights()
  model$layers[[w]]$set_weights(source_weights)
}

classifier <- art$estimators$classification$KerasClassifier(model=model, clip_values = c(0,1))

all_scores <- list()
for(i in 1:dim(x_test)[1]){
  x_single_sample <- x_test[i,,,,]
  dim(x_single_sample) <- c(dim(x_single_sample), 1)
  clever_score <- art$metrics$clever_u(classifier, x_single_sample, 
                                     nb_batches = 50L, batch_size = 10L, 
                                     radius = 10L, norm = 2)
  all_scores <- append(all_scores, clever_score)
}
