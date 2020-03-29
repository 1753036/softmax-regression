library(ramify)

load_images_file <- function(filename) {
  f <- file(filename, 'rb')
  magic_number <- readBin(f, 'integer', n=4, size=1, endian="big")
  cat(magic_number, '\n')
  n <- readBin(f, 'integer', n=1, size=4, endian="big")
  n_row <- readBin(f, 'integer', n=1, size=4, endian="big")
  n_col <- readBin(f, 'integer', n=1, size=4, endian="big")
  cat(n, n_row, n_col)
  data <- readBin(f, 'integer', n=n * n_row * n_col, size=1, endian="big", signed=FALSE)
  close(f)
  data.frame(matrix(data, ncol=n_row*n_col, byrow=TRUE))
}

train_x <- load_images_file("data/train-images-idx3-ubyte")
test_x <- load_images_file("data/t10k-images-idx3-ubyte")

load_labels_file <- function(filename) {
  f <- file(filename, 'rb')
  magic_number <- readBin(f, 'integer', n=4, size=1, endian="big")
  cat(magic_number, '\n')
  n <- readBin(f, 'integer', n=1, size=4, endian="big")
  cat(n, '\n')
  data <- readBin(f, 'integer', n=n, size=1, endian="big", signed=FALSE)
  close(f)
  data
}

train_y <- as.factor(load_labels_file("data/train-labels-idx1-ubyte"))
test_y <- as.factor(load_labels_file("data/t10k-labels-idx1-ubyte"))

show_digit <- function(data_f, col) {
  arr <- as.matrix(data_f[col,][-785])
  mat <- matrix(arr, nrow=28)[, 28:1]
  image(mat, axes=FALSE, col=grey(seq(0, 1, length=254)))
  cat(as.integer(data_f$y[col]) - 1)
}

onehot_encode <- function(Y) {
  cats <- unique(Y)
  num_col <- length(cats)
  num_row <- length(Y)
  Y_enc <- matrix(rep(0, num_col * num_row), ncol = num_col, byrow=TRUE)
  for (i in 1:num_row) {
    Y_enc[i, Y[i]] <- 1
  }
  
  return (Y_enc)
}

softmax <- function(Z) {
  e_Z <- exp(Z)
  return(e_Z / colSums(e_Z))
}

input_net <- function(W, Z) {
  return(Z %*% W)
}

make_weights <- function(num_features, num_classes) {
  v <- runif(num_features * num_classes)
  return(matrix(v, ncol=num_classes, byrow=TRUE))
}

softmax_regression <- function(input_x, input_y, learning_rate, epochs, tol) {
  num_features = ncol(input_x)
  num_classes = ncol(input_y)
  W = make_weights(num_features, num_classes)
  N = nrow(input_x)
  print(N)
  for (temp in 1:epochs) {
    for (i in 1:N) {
      X_i <- input_x[i,]
      Y_i <- input_y[i,]
      I_i <- input_net(W, X_i)
      A_i <- softmax(t(I_i))
      W_new <- W + learning_rate * (X_i %*% t(Y_i - A_i))
      if (norm(W_new - W) < tol) {
        return (W)
      }
      
      W <- W_new
    }
    cat("epochs ", temp, "/", epochs, '\n')
  }
  return(W)
}

softmax_predict <- function(W, X) {
  A <- X %*% W
  return(argmax(softmax(A)) - 1)
}

mat_train_x <- matrix(as.matrix(train_x), ncol=784) / 255
enc_train_y <- onehot_encode(train_y)
mat_test_x <- matrix(as.matrix(test_x), ncol=784) / 255
enc_train_x <- onehot_encode(test_y)

model <- softmax_regression(mat_train_x, enc_train_y, 0.01, 20, 5e-9)
test_pred <- softmax_predict(model, mat_test_x)
train_pred <- softmax_predict(model, mat_train_x)
cat("test set accurate: ", sum(test_pred == test_y) / 10000, '\n')
cat("train set accurate: ", sum(train_pred == train_y) / 60000, '\n')

