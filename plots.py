from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import arviz as az
from itertools import cycle



def cm_plot(y_test, y_pred_class, model_label, y_classes, save_path=None):
    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap="viridis", xticklabels=y_classes, yticklabels=y_classes)
    plt.title(f"{model_label} Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()




# NOTE inverse transform tfidf a le a forest plot pro n nevyznamnejsich features v kazde tride
def forest_plot_n_features(idata, tfidf, le, top_n=10, save_path=None):
    """
    Vykreslí Forest Plot pro N nejvýznamnějších slov pro každou třídu.
    """
    # 1. Získání dat z modelu
    # Zprůměrujeme přes řetězce a iterace -> dostaneme (n_features, n_classes)
    posterior_mean = idata.posterior["betas"].mean(dim=["chain", "draw"]).values

    # Získáme HDI (intervaly spolehlivosti) pro "tykadla" grafu
    # Výsledek má tvar (n_features, n_classes, 2)
    hdi = az.hdi(idata)["betas"].values

    # 2. Získání popisků
    feature_names = tfidf.get_feature_names_out()  # Slova
    class_names = le.classes_  # Třídy (Negative, Neutral, Positive)

    # Barvy pro třídy (Negative=Červená, Neutral=Šedá, Positive=Zelená)
    colors = ['#d62728', '#7f7f7f', '#2ca02c']

    # 3. Vykreslení
    fig, axes = plt.subplots(1, len(class_names), figsize=(18, 8), sharey=False)

    if len(class_names) == 1: axes = [axes]  # Fix pro případ jen 1 třídy

    for i, class_label in enumerate(class_names):
        ax = axes[i]

        # Vybereme váhy pro aktuální třídu
        class_betas = posterior_mean[:, i]
        class_hdi = hdi[:, i, :]

        # Získáme indexy seřazené od největšího po nejmenší
        # Chceme slova, která nejvíce PŘISPÍVAJÍ k dané třídě (nejvyšší kladná čísla)
        sorted_indices = np.argsort(class_betas)[::-1][:top_n]

        # Vytáhneme data pro top N
        top_words = feature_names[sorted_indices]
        top_means = class_betas[sorted_indices]
        top_hdis = class_hdi[sorted_indices]

        # Vykreslení (vypadá jako Forest Plot)
        # Y osa = slova, X osa = hodnota bety
        y_pos = np.arange(top_n)

        # Tečka (průměr)
        ax.scatter(top_means, y_pos, color=colors[i], s=100, zorder=3)

        # Čára (HDI interval)
        # errorbar chce relativní chybu, my máme absolutní limity, musíme přepočítat
        x_err = np.array([top_means - top_hdis[:, 0], top_hdis[:, 1] - top_means])
        ax.errorbar(top_means, y_pos, xerr=x_err, fmt='none', ecolor=colors[i], alpha=0.6, capsize=5, zorder=2)

        # Čára nuly (referenční)
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)

        # Popisky
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words, fontsize=12)
        ax.invert_yaxis()  # Aby bylo nejlepší nahoře
        ax.set_title(f"Class: {class_label}", fontsize=14, color=colors[i], fontweight='bold')
        ax.set_xlabel("Beta Coefficient")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()




def multiclass_roc_plot(y_test, y_score, classes, model_name, save_path=None):
    """
    Vykreslí ROC křivky pro každou třídu (One-vs-Rest).
    y_test: Skutečné třídy (integery nebo stringy)
    y_score: Pravděpodobnosti (output z predict_proba)
    classes: Seznam názvů tříd (např. le.classes_)
    """
    # 1. Binarizace y_test (převede 0,1,2 na [1,0,0], [0,1,0], [0,0,1])
    # Musíme vědět, kolik máme tříd. Předpokládáme, že y_test už jsou čísla (0, 1, 2)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # 2. Výpočet ROC a AUC pro každou třídu
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 3. Vykreslení
    plt.figure(figsize=(8, 6))

    # Barvy pro jednotlivé třídy
    colors = cycle(['blue', 'red', 'green'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    # Diagonála (náhoda)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC: {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()




def mcmc_diag_plots(idata, post_pred, save_path, n_features=15, n_pp_samples=100):
    # TRACE
    axes = az.plot_trace(idata, var_names=["intercept"])
    axes[0, 0].set_title(f"Overlay")
    axes[0, 1].set_title(f"Trace")
    plt.suptitle(f"Intercept", fontsize=16)
    plt.savefig(save_path / "trace_intercept.png")
    plt.close()

    # AUTOCORRELATION
    az.plot_autocorr(idata, var_names=["intercept"])
    plt.suptitle(f"Intercept - Autocorrelation", fontsize=26)
    plt.savefig(save_path / "autocorr_intercept.png")
    plt.close()

    for i in range(n_features):
        # TRACE
        axes = az.plot_trace(idata, var_names=["betas"], coords={"betas_dim_0": [i]})
        axes[0, 0].set_title(f"Overlay")
        axes[0, 1].set_title(f"Trace")
        plt.suptitle(f"Beta {i}", fontsize=16)
        plt.savefig(save_path / f"trace_beta_{i}.png")
        plt.close()

        # AUTOCORRELATION
        az.plot_autocorr(idata, var_names=["betas"], coords={"betas_dim_0": [i]})
        plt.suptitle(f"Beta {i} - Autocorrelation", fontsize=26)
        plt.savefig(save_path / f"autocorr_beta_{i}.png")
        plt.close()


    # PPC
    total_samples = post_pred.posterior_predictive.sizes['chain'] * post_pred.posterior_predictive.sizes['draw']
    # Použijeme buď požadovaných n_pp_samples, nebo maximum dostupných, pokud jich je méně
    safe_samples = min(n_pp_samples, total_samples)

    az.plot_ppc(post_pred, num_pp_samples=safe_samples, kind='kde')
    plt.savefig(save_path / "ppc_train.png")
    plt.close()