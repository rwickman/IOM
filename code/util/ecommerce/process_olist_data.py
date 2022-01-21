import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN, KMeans
import json
from sklearn.metrics import silhouette_score
from tqdm import tqdm

TRAIN_SPLIT = 0.8

SAVE_LOC = "data/olist_data/"
OLIST_PATH = "code/util/ecommerce/archive/"

def split_df(df):
    """Split DataFrame into train, validation, and testing set."""
    num_train_rows = int(TRAIN_SPLIT * len(df))

    train_loc_df, test_loc_df = df[:num_train_rows], df[num_train_rows:]
    
    num_test_rows = len(test_loc_df) // 2
    test_loc_df, val_loc_df = test_loc_df[num_test_rows:], test_loc_df[:num_test_rows]

    return train_loc_df, val_loc_df, test_loc_df

def save_customer_loc(save_loc):
    """Get customer locations.

    Returns:
        the dataframe of locations for the train and test locations
    """
    loc_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_geolocation_dataset.csv"))
    # print("loc_df", loc_df)
    cust_df = pd.read_csv("code/util/ecommerce/archive/olist_customers_dataset.csv")
    cust_locs_df = loc_df[
            loc_df["geolocation_zip_code_prefix"].isin(cust_df["customer_zip_code_prefix"]) & 
            loc_df["geolocation_city"].isin(cust_df["customer_city"]) &
            loc_df["geolocation_state"].isin(cust_df["customer_state"])]
    

    # cust_locs = loc_df[["geolocation_lat", "geolocation_lng"]]
    

    # Get only geolocation and shuffle them
    cust_locs_df = cust_locs_df[["geolocation_lat", "geolocation_lng"]]
    #train_loc_df, val_loc_df, test_loc_df = split_df(cust_locs_df)

    cust_locs_df.to_csv(os.path.join(save_loc, "train_loc.csv"))
    # val_loc_df.to_csv(os.path.join(save_loc, "val_loc.csv"))
    # test_loc_df.to_csv(os.path.join(save_loc, "test_loc.csv"))
    # train_loc_df = train_loc_df.to_numpy()
    # val_loc_df = val_loc_df.to_numpy()
    # test_loc_df = test_loc_df.to_numpy()


    # # Plot them
    # fig, ax = plt.subplots(3)

    # ax[0].scatter(train_loc_df[:, 0], train_loc_df[:, 1])
    # ax[1].scatter(val_loc_df[:, 0], val_loc_df[:, 1])
    # ax[2].scatter(test_loc_df[:, 0], test_loc_df[:, 1])

    # for i in range(3):
    #     ax[i].set(xlabel="Latitude", ylabel="Longitude")
    #     ax[i].set_title("Customer Locations", fontsize=24)
    #     #plt.subplots_adjust(left=0.15, bottom=0.15)
    #     ax[i].set_xticks(np.arange(-40, 60, 10))
    #     ax[i].set_yticks(np.arange(-100, 40, 20))


    # return train_loc_df, val_loc_df, test_loc_df
    


def save_orders(PID_threshold=10):
    """Get the orders for training, testing, and validation.
    
    Args:
        PID_threshold: number of times PID was used.
    """
    # 1. Filter out orders that have a PID that was used less than a threshold given number of times
    # 2. Get distrubtion 
    # 2. Use first 80% as train data 
    orders_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_orders_dataset.csv"))
    order_items_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_order_items_dataset.csv"))

    pid_count = order_items_df["product_id"].value_counts()
    
    # Get only orders that have a PID that was used at least a certian number of times 
    filtered_order_items_df = order_items_df[order_items_df.product_id.isin(pid_count.index[pid_count.ge(PID_threshold)])]
    sku_ids_dict = {}
    p_ids = filtered_order_items_df["product_id"].unique()

    for i in range(len(p_ids)):
        sku_ids_dict[p_ids[i]] = i

    # Get the new PID count 
    # pid_count = filtered_order_items_df["product_id"].value_counts()
    # pid_count = pid_count.to_numpy()
    
    # Sample using weight of PID
    #sample_prob = np.random.choice(np.arange(len(pid_count)), p=pid_count/pid_count.sum())
    
    # Split into train
    order_ids = filtered_order_items_df["order_id"].unique()

    #print(len(filtered_order_items_df))
    #print(len(order_ids))
    rand_state = np.random.RandomState(12)
    rand_state.shuffle(order_ids)

    train_order_ids, val_order_ids, test_order_ids = split_df(order_ids)
    print("filtered_order_items_df", filtered_order_items_df)
    train_orders = filtered_order_items_df[filtered_order_items_df["order_id"].isin(train_order_ids)]
    val_orders = filtered_order_items_df[filtered_order_items_df["order_id"].isin(val_order_ids)]
    print("val_orders", val_orders)
    test_orders = filtered_order_items_df[filtered_order_items_df["order_id"].isin(test_order_ids)]

    # Save the orders
    # train_orders.to_csv(os.path.join(SAVE_LOC, "orders/train_orders.csv"))
    # val_orders.to_csv(os.path.join(SAVE_LOC, "orders/val_orders.csv"))
    # test_orders.to_csv(os.path.join(SAVE_LOC, "orders/test_orders.csv"))

    # Map the product IDS to numbers

    with open(os.path.join(SAVE_LOC, "pid_to_sku_id.json"), "w") as f:
        json.dump(sku_ids_dict, f) 

    train_orders = train_orders.replace({"product_id": sku_ids_dict})
    val_orders = val_orders.replace({"product_id": sku_ids_dict})
    test_orders = test_orders.replace({"product_id": sku_ids_dict})

    
    train_orders.to_csv(os.path.join(SAVE_LOC, "orders/train_orders.csv"))
    val_orders.to_csv(os.path.join(SAVE_LOC, "orders/val_orders.csv"))
    test_orders.to_csv(os.path.join(SAVE_LOC, "orders/test_orders.csv"))

    
    

    # print(len(train_orders.order_id.unique()))
    # print("len(train_order_ids): ", len(train_orders))
    # print("len(test_orders[order_id].unique())", len(test_orders["order_id"].unique()))
    # print("len(val_orders[order_id].unique())", len(val_orders["order_id"].unique()))
    # print("AVG VALUE COUNT: ", train_orders["product_id"].value_counts().mean())
    # print("AVG VALUE COUNT: ", val_orders["product_id"].value_counts().mean())
    # print("AVG VALUE COUNT: ", test_orders["product_id"].value_counts().mean())

    # Get the average demand
    # print(train_orders["order_id"].value_counts().mean())

def calculate_WSS(points, kmax, kmin=8):
    sse = []
    sil = []
    for k in range(kmin, kmax+1):
        print(k)
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in tqdm(range(len(points))):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
            
        sse.append(curr_sse)
        labels = kmeans.labels_
        sil.append(silhouette_score(points, labels, metric = 'euclidean', sample_size=20000))
    
    return sse, sil

def save_inventory_nodes(num_inv_nodes=4):
    loc_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_geolocation_dataset.csv"))
    seller_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_sellers_dataset.csv"))
    seller_locs_df = loc_df[
            loc_df["geolocation_zip_code_prefix"].isin(seller_df["seller_zip_code_prefix"]) & 
            loc_df["geolocation_city"].isin(seller_df["seller_city"]) &
            loc_df["geolocation_state"].isin(seller_df["seller_state"])]
    
    seller_locs = seller_locs_df[["geolocation_lat", "geolocation_lng"]].to_numpy()
    # print(seller_locs.shape)
    sse, sil = calculate_WSS(seller_locs, 15, 12)
    fig, axs = plt.subplots(2)
    axs[0].plot(sse)
    axs[1].plot(sil)

    plt.show()
    
    # Create seller locations us DBSCAN
    clustering = KMeans(n_clusters=13).fit(seller_locs)
    
    inv_node_locs = clustering.cluster_centers_.tolist()


    
    with open(os.path.join(SAVE_LOC, "location/inv_node_locs.json"), "w") as f:
        json.dump(inv_node_locs, f)
    plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1])

    #plt.scatter(seller_locs[:, 0], seller_locs[:, 1])
    plt.show()

def create_inv_stock(orders_csv, inv_loc_json, is_test=False):
    """Create the stock for the inventory nodes for test/validation set."""
    loc_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_geolocation_dataset.csv"))
    seller_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_sellers_dataset.csv"))
    orders_df = pd.read_csv(orders_csv)
    
    # Get sellers in the orders
    order_sellers_df = seller_df[seller_df["seller_id"].isin(orders_df["seller_id"])]
    
    # Get the locations of the sellers
    order_sellers_loc_df = loc_df[
        loc_df["geolocation_zip_code_prefix"].isin(order_sellers_df["seller_zip_code_prefix"]) & 
        loc_df["geolocation_city"].isin(order_sellers_df["seller_city"]) &
        loc_df["geolocation_state"].isin(order_sellers_df["seller_state"])]


    # Get the mean of the locations, since there are duplicates in the location dataset
    order_sellers_loc_df = order_sellers_loc_df.groupby(
        ["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"], as_index=False).mean()
        
    # Rename for convenience of joining 
    order_sellers_loc_df.columns = ["seller_zip_code_prefix", "seller_city", "seller_state",
        "geolocation_lat", "geolocation_lng"]
    
    # Add the geolocations to the sellers DataFrame
    order_sellers_df = order_sellers_df.merge(order_sellers_loc_df, how="left", on=["seller_zip_code_prefix", "seller_city", "seller_state"])
    
    # Fix NA as some columns values were entered wrong for sellers so match could not be found
    avg_loc_state = order_sellers_loc_df.groupby(["seller_state"], as_index=False).mean()
    na_rows = order_sellers_df["geolocation_lat"].isna()

    order_sellers_df.loc[na_rows ,"geolocation_lat"]  = order_sellers_df[na_rows].merge(
        avg_loc_state, on="seller_state", how="left")["geolocation_lat_y"].tolist()
    order_sellers_df.loc[na_rows ,"geolocation_lng"]  = order_sellers_df[na_rows].merge(
        avg_loc_state, on="seller_state", how="left")["geolocation_lng_y"].tolist()
    
    inv_locs = []
    with open(inv_loc_json) as f:
        inv_locs = json.load(f)
    
    # with open(os.path.join(SAVE_LOC, "location/pid_to_sku_id.json")) as f:
    #     sku = json.load(f)

    # Map each seller to inventory node
    inv_stock_dict = {}
    
    dist_fn = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
    for idx, row in order_sellers_df.iterrows():
        min_dist = None
        min_inv_idx = None
        
        for inv_idx in range(len(inv_locs)):
            cur_dist = dist_fn(inv_locs[inv_idx], [row["geolocation_lat"], row["geolocation_lng"]])
            if min_dist is None or cur_dist < min_dist:
                
                min_dist = cur_dist
                min_inv_idx = inv_idx
        
        # Get the inventory for this seller
        if min_inv_idx not in inv_stock_dict:
            inv_stock_dict[min_inv_idx] = {}
        
        
        for sku_id in orders_df[orders_df["seller_id"] == row["seller_id"]]["product_id"]:
            if sku_id not in inv_stock_dict[min_inv_idx]:
                inv_stock_dict[min_inv_idx][sku_id] = 0
            inv_stock_dict[min_inv_idx][sku_id] += 1
        # inv_stock_dict[min_inv_idx] = 
        #seller_node_dict[row["seller_id"]] = min_inv_idx
    
    # Create the inventory for all the nodes

    #print(seller_node_dict)
    if is_test:
        inv_stock_save_file = os.path.join(SAVE_LOC, "inv_stock/test_inv_node_stock.json")
    else:
        inv_stock_save_file = os.path.join(SAVE_LOC, "inv_stock/val_inv_node_stock.json")

    with open(inv_stock_save_file, "w") as f:
        json.dump(inv_stock_dict, f)

    
def create_orders(order_csv, is_test=False):
    cust_df = pd.read_csv("code/util/ecommerce/archive/olist_customers_dataset.csv")
    order_df = pd.read_csv(order_csv)
    loc_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_geolocation_dataset.csv"))
    olist_order_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_orders_dataset.csv"))
    #order_items_df = pd.read_csv(os.path.join(OLIST_PATH, "olist_order_items_dataset.csv"))
    
    # Get the customers in the orders
    olist_order_df = olist_order_df[olist_order_df["order_id"].isin(order_df["order_id"])]
    cust_df = cust_df[cust_df["customer_id"].isin(olist_order_df["customer_id"])]
    
    
    # Get the locations of the customers
    cust_locs = loc_df[
            loc_df["geolocation_zip_code_prefix"].isin(cust_df["customer_zip_code_prefix"]) & 
            loc_df["geolocation_city"].isin(cust_df["customer_city"]) &
            loc_df["geolocation_state"].isin(cust_df["customer_state"])]



    # Get the mean of the locations, since there are duplicates in the location dataset
    cust_locs = cust_locs.groupby(
        ["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"], as_index=False).mean()

    
    # Rename for convenience of joining 
    cust_locs.columns = ["customer_zip_code_prefix", "customer_city", "customer_state",
        "geolocation_lat", "geolocation_lng"]
    
    
    # Add the geolocations to the customer DataFrame
    cust_df = cust_df.merge(cust_locs, how="left", on=["customer_zip_code_prefix", "customer_city", "customer_state"])
    
    # Fix NA as some columns values were entered wrong for customers so match could not be found
    avg_loc_state = cust_locs.groupby(["customer_state"], as_index=False).mean()
    na_rows = cust_df["geolocation_lat"].isna()

    cust_df.loc[na_rows ,"geolocation_lat"]  = cust_df[na_rows].merge(
        avg_loc_state, on="customer_state", how="left")["geolocation_lat_y"].tolist()
    cust_df.loc[na_rows ,"geolocation_lng"]  = cust_df[na_rows].merge(
        avg_loc_state, on="customer_state", how="left")["geolocation_lng_y"].tolist()

    # Added location to cust_df, now need to get products in order
    # For each customer, get the order_IDS from 
    print("cust_df", cust_df)\
    # Contains order_id and product_id
    print("order_df", order_df)
    # Contains order_id and customer_id
    print("olist_order_df", olist_order_df)
    
    cust_product_ids = []
    for idx, row in cust_df.iterrows():
        print("idx", idx)
        # Get the customer's order_ids
        order_ids = olist_order_df[olist_order_df["customer_id"] == row["customer_id"]]["order_id"].tolist()

        cur_product_ids = []
        for order_id in order_ids:
            cur_product_ids += order_df[order_df["order_id"] == order_id]["product_id"].tolist()

        cust_product_ids.append(cur_product_ids)
    
    if is_test:
        cust_orders_json = os.path.join(SAVE_LOC, "orders/test_cust_orders.json")
    else:
        cust_orders_json = os.path.join(SAVE_LOC, "orders/val_cust_orders.json")

    with open(cust_orders_json, "w") as f:
        json.dump(cust_product_ids, f)

        
    
        

    # print("cust_df", cust_df["geolocation_lat"].isna().sum())
    # print("cust_df", cust_df["geolocation_lng"].isna().sum())


    
    # print("order_df", order_df.columns)
    # print("order_items_df", order_items_df.columns)

create_orders("data/olist_data/orders/val_orders.csv")
#save_customer_loc("data/olist_data/location")
# save_orders()
# save_inventory_nodes()

# create_inv_stock("data/olist_data/orders/val_orders.csv", "data/olist_data/location/inv_node_locs.json")

# create_inv_stock("data/olist_data/orders/test_orders.csv", "data/olist_data/location/inv_node_locs.json", is_test=True)