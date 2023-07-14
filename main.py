import numpy as np
from io_data import read_data
from io_data import write_data


def gaussianPDF(data, mean, covariance):
    d = data.shape[1]
    # Using gaussian pdf formula to ge denom and exponent then returning np.exp(exponent)/denom
    denominator = np.sqrt(((2 * np.pi) ** d) * np.linalg.det(covariance))
    x_minus_mean = data - mean
    exponent = -(1 / 2) * np.sum(np.dot(x_minus_mean, np.linalg.inv(covariance)) * x_minus_mean, axis=1)

    return np.exp(exponent) / denominator


def logLikelihoodInitial(data, mean, covariance, weights):
    d = data.shape[1]
    n = data.shape[0]
    k = weights.shape[0]
    pdfs = np.zeros((n, k))

    # Get all the pdfs for all the k's
    for i in range(k):
        pdfs[:, i] = weights[i] * gaussianPDF(data, mean[i], covariance[i])
    # log likelihood is  equal to sum of the log of the sum of the pdfs
    # aka the code below
    log_likelihood = np.sum(np.log(np.sum(pdfs, axis=1)))
    return log_likelihood


def kmeansInitialization(data, k, initial):
    # Choose two random centroids to start the kmeans initialization
    n = data.shape[0]
    d = data.shape[1]
    mean = np.array(initial)

    assignments = np.zeros(n, dtype=int)  # Used for filtering data

    # Run it a few times to get it more accurate NOTE JK decreased it to one so it wouldnt overfit
    iterations = 1

    for i in range(iterations):
        # Getting the minimum distances and assigning groups
        distances = np.linalg.norm(data[:, np.newaxis, :] - mean, axis=2)
        assignments = np.argmin(distances, axis=1)
        for j in range(k):
            mean[j, :] = np.mean(data[assignments == j, :], axis=0)

    covariance = np.zeros((k, d, d))
    for i in range(k):
        # Get the variance of all the relevant data
        variance_lst = data[assignments == i, :] - mean[i, :]
        covariance[i, :, :] = np.dot(variance_lst.T, variance_lst) / (n - 1)

    return mean, covariance, (np.array([len(data[assignments == 0]), len(data[assignments == 1])]) / n)


def gmm(file_name, initial):
    data, image = read_data(file_name, False)

    img_data = np.array(data[:, 2:])

    n = img_data.shape[0]
    d = img_data.shape[1]
    # Run k means for num_clusters to initialize covariances
    # I need a mean/mu 2 by 3 (k by d). one average for each of the clusters
    # covariance / sigma 2 by 3 by 3(k by d by d) one  3 by 3 matrix for each k
    # p/ percentage assigned to k one n_k / n for each l
    # count number of assigned to k
    k = 2

    mean, covariance, weights = kmeansInitialization(img_data, k, initial)

    # Now get the initial log likelihood for comparison of convergence
    prev_ll = -np.inf
    log_likelihood = logLikelihoodInitial(img_data, mean, covariance, weights)

    # Starting iterative loop until convergence
    while True:
        print('Starting next iteration\nLL Dif: ', log_likelihood - prev_ll)
        # e step get the responsibilities /soft assignments
        responsibilities = np.zeros((n, k))  # Responsibilities will hold all the updated soft assignments
        # go through all the samples with each weight

        for i in range(k):
            responsibilities[:, i] = weights[i] * gaussianPDF(img_data, mean[i], covariance[i])

        for i in range(n):
            responsibilities[i] /= np.sum(responsibilities[i])

        # Onto the M step teehhee
        for i in range(k):  # Iterate through each var for k
            # update all the parameters by formulas
            sum = np.sum(responsibilities[:, i])
            weights[i] = sum / n
            mean[i] = np.sum(responsibilities[:, i, np.newaxis] * img_data, axis=0) / sum
            x_minus_mean = img_data - mean[i]
            covariance[i] = np.dot(responsibilities[:, i] * x_minus_mean.T, x_minus_mean) / sum

        prev_ll = log_likelihood
        log_likelihood = np.sum(np.log(np.sum(responsibilities, axis=1)))
        if log_likelihood - prev_ll < 1e-6:
            print(log_likelihood - prev_ll)
            break

    get_labels = np.argmax(responsibilities, axis=1)
    data_out1, data_out2 = getImageData(get_labels, data, False)
    data_mask, d = getImageData(get_labels, data)
    return data_out1, data_out2, data_mask


def getImageData(labels, data, mask=True):
    white = (100, 0, 0)
    black = (0, 0, 0)
    if mask:
        for i in range(len(data)):
            if labels[i]:
                data[i, 2] = white[0]
                data[i, 3] = white[1]
                data[i, 4] = white[2]
            else:
                data[i, 2] = black[0]
                data[i, 3] = black[1]
                data[i, 4] = black[2]
        return data, data
    else:
        data1 = data.copy()
        data0 = data.copy()
        for i in range(len(data)):
            if labels[i]:
                data0[i, 2] = black[0]
                data0[i, 3] = black[1]
                data0[i, 4] = black[2]
            else:
                data1[i, 2] = black[0]
                data1[i, 3] = black[1]
                data1[i, 4] = black[2]
        return data0, data1


def __main__():
    data_names = ['cow', 'fox', 'owl', 'zebra']
    initials = [[[50, 0, 0], [46, -22, 40]],
               [[46, 13, 34], [46, -22, 40]],
               [[95, -1, 3], [70, 2, 22]],
               [[50, 0, 0], [64, -20, 46]]
               ]
    for i in range(len(data_names)):

        out1, out2, out_mask = gmm('data/' + data_names[i] + '.txt', initials[i])

        write_data(out1, 'data/out1_' + data_names[i] + '.txt')
        write_data(out2, 'data/out2_' + data_names[i] + '.txt')
        write_data(out_mask, 'data/outmask_' + data_names[i] + '.txt')

        data, image = read_data("data/outmask_" + data_names[i] + ".txt", False, False, True, "data/mask_" + data_names[i] + ".jpg")
        data, image = read_data('data/out2_' + data_names[i] + '.txt', False, False, True, "data/out1_" + data_names[i] + ".jpg")
        data, image = read_data('data/out1_' + data_names[i] + '.txt', False, False, True, "data/out2_" + data_names[i] + ".jpg")



__main__()
