import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
    # print(val.mean()m val.std())

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
    
    ax.hist(labels, bins=len(levels), color="blue")
    values, counts = np.unique(p_ids.to_numpy(), return_counts=True)
    #sns.barplot(x=values, y=counts)
    ax.set(xlabel="Product ID", ylabel="Number of times in Order")
    ax.set_title("Product Histogram", fontsize=24)
    plt.savefig("figures/product_hist.png")
    _, counts = np.unique(p_ids.to_numpy(), return_counts=True)
    print(counts.mean(), counts.std(), counts.min(), counts.max(), np.median(counts))
    print(np.quantile(counts, 0.25), np.quantile(counts, 0.5), np.quantile(counts, 0.75))
    print(stats.mode(counts))

def estimate_shipping_cost():
    prod_df = pd.read_csv("archive/olist_products_dataset.csv")
    prod_df = prod_df.set_index("product_id")
    order_df = pd.read_csv("archive/olist_order_items_dataset.csv")
    # freight_value, product_id
    # product_weight_g, product_length_cm, product_height_cm, product_width_cm
    # Consider getting distance to customer from customer to seller if LR value is not high enough
    # order_df
    
    # Remove NAN rows
    nan_prod_ids = prod_df[prod_df["product_length_cm"].isna()].index.to_numpy()
    order_df = order_df[~order_df["product_id"].isin(nan_prod_ids)]
    # Remove rows with zero freight values
    order_df = order_df[order_df["freight_value"] != 0]

    p_ids = order_df["product_id"].to_numpy()
    x = prod_df.loc[p_ids][["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]].to_numpy()
    freight_values = order_df["freight_value"].to_numpy()
    
    print("NUM ZEROS:", len(freight_values) - np.count_nonzero(freight_values))
    x_train, x_test, y_train, y_test = train_test_split(x, freight_values, test_size=0.2)

    #print("x", x)
    print("freight_values", freight_values.mean(), freight_values.std(), freight_values.min(), freight_values.max())
    print("Quartiles:", np.quantile(freight_values, 0.25), np.quantile(freight_values, 0.5), np.quantile(freight_values, 0.75))
    reg = LinearRegression().fit(x_train, y_train)
    print("Coefficients", reg.coef_)

    print("Intercept", reg.intercept_)
    print(reg.score(x_train, y_train))
    print(reg.score(x_test, y_test))
    y_pred = reg.predict(x_test)
    print("PREDICTIONS", y_pred)
    print("ACTUAL", y_test)
    diff = y_pred - y_test
    fig, ax = plt.subplots(1)
    ax.set(xlabel="Product Weight (g)", ylabel="Freight Value")
    ax.set_title("Cost vs Weight", fontsize=24)
    print("freight_values", freight_values)
    ax.scatter(x[:, 0], freight_values)
    # plt.show()
    plt.savefig("figures/cost_vs_weight.png")

    fig, ax = plt.subplots(1)
    ax.set(xlabel="Product Length (cm)", ylabel="Freight Value")
    ax.set_title("Cost vs Length", fontsize=24)
    ax.scatter(x[:, 1], freight_values)
    # plt.show()
    plt.savefig("figures/cost_vs_length.png")
    # print("DIFF", diff)
    # print("AVG DIFF", diff.mean())


def arrival_time():
    # Get the order_purchase_time from olist_orders_dataset
    # 
    orders_df = pd.read_csv("archive/olist_orders_dataset.csv")
    order_times = pd.to_datetime(orders_df["order_purchase_timestamp"].sort_values())
    time_deltas = (order_times - order_times.shift()).fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
    
    # Plot histogram of arrival time
    fig, ax = plt.subplots(1)
    ax.hist(np.log(time_deltas.to_numpy()+1))
    ax.set(xlabel="log(seconds)", title="Arrival Time Histogram")
    plt.show()
    print(time_deltas)
    print(time_deltas[1:].mean())
    print(time_deltas[1:].std())
    print(time_deltas[1:].max())
    print(time_deltas[1:].min())
    print("Quartiles:", np.quantile(time_deltas, 0.25), np.quantile(time_deltas, 0.5), np.quantile(time_deltas, 0.75))


def interarrival_time(pid_threshold=1):
    # Need to get what times each specific product has arrived

    # 1. For each order_id in olist_products_dataset.csv, need to get the time it was ordered from olist_orders_dataset.csv"

    orders_df = pd.read_csv("archive/olist_orders_dataset.csv")
    #print(orders_df)
    order_items_df = pd.read_csv("archive/olist_order_items_dataset.csv")
    p_ids = order_items_df["product_id"].astype("category")
    #print(p_ids)
    labels, levels = pd.factorize(order_items_df["product_id"])

    values, counts = np.unique(p_ids.to_numpy(), return_counts=True)
    #print(values, counts)
    pid_count = order_items_df["product_id"].value_counts()
    # print(pid_count.index[pid_count.ge(pid_threshold)])
    order_id_counts = order_items_df["order_id"].value_counts()
    d = {}
    for i in range(1, 22):
        num_orders = len(order_id_counts.index[order_id_counts.eq(i)].value_counts()) 
        if num_orders > 0:
            d[i] = num_orders 

    order_id_hist = np.array(list(d.items()))
    # Get the number of orders left at each threshold
    for i in range(1, 7):
        print(i, order_id_hist[i-1:, 1].sum())

    #print("order_id counts", order_id_counts.index[order_id_counts.eq(4)].value_counts())
    
    
    #print(order_items_df[order_items_df.product_id.isin(pid_count.index[pid_count.ge(pid_threshold)])])

    
    
    #df = pd.merge(order_items_df, orders_df, on="order_id")
    #print(df.sort_values(["product_id", "order_purchase_timestamp"]))
    
    # p_ids = order_df["product_id"].astype("category")
    # print(p_ids)
    # labels, levels = pd.factorize(order_df["product_id"])
    # print(labels)

    #print(order_times - order_times.shift())


        

#plot_order_hist(df)
#plot_customer_loc()
# get_products()
#arrival_time()
interarrival_time()
#estimate_shipping_cost()
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

