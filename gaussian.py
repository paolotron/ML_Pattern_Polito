import numpy as np
import matplotlib.pyplot as plt
import ml_p.probability as p

if __name__ == "__main__":
    gaus = np.load("datasets/XGau.npy")
    plt.hist(gaus, bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    ll_samples = p.GAU_pdf(XPlot, 0, 2)
    plt.plot(XPlot, ll_samples)
    plt.title("normal density vs the data")
    plt.show()

    print("The likelihood is: ", ll_samples.prod())

    mu = np.mean(gaus)
    v = np.var(gaus)
    log_gaus = p.GAU_logpdf(gaus, mu, v)
    plt.hist(gaus, bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    log_samples = p.GAU_logpdf(XPlot, mu, v)
    plt.plot(XPlot, np.exp(log_samples))
    plt.title("normal density with estimation of mu and v vs The Data")
    plt.show()
    print(sum(log_gaus))

    XND = np.load("Solutions/XND.npy")
    mu = np.load("Solutions/muND.npy")
    C = np.load("Solutions/CND.npy")
    pdfSol = np.load("Solutions/llND.npy")
    pdfGau = p.GAU_ND_logpdf(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).mean())


