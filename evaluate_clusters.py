import os

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from utils import get_logger, load_distances, load_embeddings

import wandb


def KNN(args):
    db_path = f"/globalscratch/ucl/irec/darimez/dino/embeddings/"
    tables = [
        os.path.join(db_path, file)
        for file in os.listdir(db_path)
        if file.endswith(".db")
    ]  # extend as needed
    logger = get_logger(args.debug)
    if not args.debug:
        run = wandb.init(
            entity="miro-unet",
            project="VLLM clustering",
            # mode="offline",
            name=f"visu",  # optional descriptive name
        )
    else:
        run = wandb
    for table in tables:
        wandb_table = [[] * 3]
        # Start a new wandb run per model
        logger.debug(table)
        ids, X, hfs, mious, mconfs, cats, supercats = load_embeddings(table)
        distances = load_distances(table.replace("embeddings", "distances"))

        indices = np.arange(len(distances))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        D_train = distances[train_idx][:, train_idx]  # square train distance matrix
        D_test = distances[test_idx][:, train_idx]  # test-to-train distances

        y_train = mious[train_idx]
        y_test = mious[test_idx]

        knn = KNeighborsRegressor(
            n_neighbors=5,
            metric="precomputed",
            weights="distance",  # optional but recommended for regression
        )
        knn.fit(D_train, y_train)
        y_pred = knn.predict(D_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("MSE:", mse, "R2:", r2)
        wandb_table[0].append(mse)
        wandb_table[1].append(r2)
        wandb_table[2].append(np.corrcoef(y_test, y_pred)[0, 1])
        wandb.log(
            {table: wandb.Table(data=wandb_table, columns=["MSE", "R2", "Correlation"])}
        )
