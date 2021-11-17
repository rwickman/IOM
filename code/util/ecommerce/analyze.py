import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

sns.set(style="darkgrid", font_scale=1.1)
DEFAULT_MARKER_SIZE = mpl.rcParams["lines.markersize"] ** 2

def plot_order_hist(df):
    # Gives the number of items ordersd
    x = df["order_item_id"].to_numpy()

    # Order item histogram
    fig, ax = plt.subplots(1)
    ax.set(xlabel="Number of Items", ylabel="Number of Orders")

    # Count unique 
    values, counts = np.unique(x, return_counts=True)
    counts = counts.tolist()

    plt.subplots_adjust(left=0.15)
    ax.bar(values, counts)
    
    # Print out values and counts
    d = {} 
    for val, count in zip(values, counts):
        d[val] = count
    print(d)

    plt.savefig("figures/num_order_item_hist.png")

def plot_customer_loc():
    loc_df = pd.read_csv("archive/olist_geolocation_dataset.csv")
    # print("loc_df", loc_df)
    cust_df = pd.read_csv("archive/olist_customers_dataset.csv")
    cust_locs = cust_df[["customer_zip_code_prefix", "customer_city", "customer_state"]].values
    # cust_loc_set = set([tuple(e) for e in cust_locs.tolist()])
    # locs = []
    # for i, row in loc_df.iterrows():
    #     if i % 512 == 0:
    #         print(i)
    #     cur_loc = (
    #         row["geolocation_zip_code_prefix"], 
    #         row["geolocation_city"],
    #         row["geolocation_state"])
    #     if cur_loc in cust_loc_set:
    #         locs.append((row["geolocation_lat"], row["geolocation_lng"]))
    # locs = np.array(locs)
    cust_locs_df = loc_df[
            loc_df["geolocation_zip_code_prefix"].isin(cust_df["customer_zip_code_prefix"]) & 
            loc_df["geolocation_city"].isin(cust_df["customer_city"]) &
            loc_df["geolocation_state"].isin(cust_df["customer_state"])]
    locs = cust_locs_df[["geolocation_lat", "geolocation_lng"]].to_numpy()
    #print(locs)
    fig, ax = plt.subplots(1)
    ax.scatter(locs[:, 0], locs[:, 1])

    ax.set(xlabel="Latitude", ylabel="Longitude")
    ax.set_title("Customer Locations", fontsize=24)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    ax.set_xticks(np.arange(-40, 60, 10))
    ax.set_yticks(np.arange(-100, 40, 20))
    plt.savefig("figures/customer_loc.png")


    # Plot seller locations
    seller_df = pd.read_csv("archive/olist_sellers_dataset.csv")
    seller_locs_df = loc_df[
            loc_df["geolocation_zip_code_prefix"].isin(seller_df["seller_zip_code_prefix"]) & 
            loc_df["geolocation_city"].isin(seller_df["seller_city"]) &
            loc_df["geolocation_state"].isin(seller_df["seller_state"])]
    
    seller_locs = seller_locs_df[["geolocation_lat", "geolocation_lng"]].to_numpy()

    fig, ax = plt.subplots(1)
    ax.scatter(seller_locs[:, 0], seller_locs[:, 1])
    
    ax.set(xlabel="Latitude", ylabel="Longitude")
    ax.set_title("Seller Locations", fontsize=24)
    ax.set_xticks(np.arange(-40, 60, 10))
    ax.set_yticks(np.arange(-100, 40, 20))

    #ax.set_xticks([int(locs[:, 0].min()), int(locs[:, 0].max())])
    #ax.set_yticks([int(locs[:, 1].min()), int(locs[:, 1].max())])
    #ax.set_yticks(np.arange(int(locs[:, 1].min()), int(locs[:, 1].max())))

    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig("figures/seller_loc.png")


def get_products():
    order_df = pd.read_csv("archive/olist_order_items_dataset.csv")
    p_ids = order_df["product_id"].astype("category")
    print(p_ids)
    labels, levels = pd.factorize(order_df["product_id"])
    print(labels)
    fig, ax = plt.subplots(1)
    
    # ax.hist(labels, bins=len(levels), color="blue")
    values, counts = np.unique(p_ids.to_numpy(), return_counts=True)
    sns.barplot(x=values, y=counts)
    ax.set(xlabel="Product ID", ylabel="Number of times in Order")
    ax.set_title("Product Histogram", fontsize=24)
    plt.savefig("figures/product_hist.png")
    _, counts = np.unique(p_ids.to_numpy(), return_counts=True)
    print(counts.mean(), counts.std(), counts.min(), counts.max(), counts.median())

#plot_order_hist(df)
#plot_customer_loc()
get_products()

# How to get customer geolocation
# 1. Get customer_zip_code_prefix from olist_customers_dataset.csv
# 2. Get geolocation_lat and geolocation_Ing from olist_geolocation_dataset.csv
# df = pd.read_csv("archive/olist_customers_dataset.csv")
# zipcodes = df["customer_zip_code_prefix"].to_numpy()
# unique_zipcodes = np.unique(zipcodes)

# How to scale scatter plot by likelihood of customer coming from location
# 1. Get customer_zip_code_prefix from olist_customers_dataset.csv
# 2. Get geolocation_lat and geolocation_Ing from olist_geolocation_dataset.csv
# 3. Count for each zipcode
# 4. Compute likelhood for each location and multiply by rcParams['lines.markersize'] ** 2.

