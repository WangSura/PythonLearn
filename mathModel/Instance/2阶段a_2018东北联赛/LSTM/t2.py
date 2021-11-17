print("....Data loading....")
print()
print('\033[4mHenry Hub Natural Gas Spot Price, Daily (Dollars per Million Btu)\033[0m')


def retrieve_time_series(api, series_ID):
    series_search = api.data_by_series(series=series_ID)
    spot_price = DataFrame(series_search)
    return spot_pricedef main():
    try:
        api_key = "....API KEY..."
        api = eia.API(api_key)
        series_ID = 'xxxxxx'
        spot_price = retrieve_time_series(api, series_ID)
        print(type(spot_price))
        return spot_price
    except Exception as e:
        print("error", e)
        return DataFrame(columns=None)


spot_price = main()
spot_price = spot_price.rename(
    {'Henry Hub Natural Gas Spot Price, Daily (Dollars per Million Btu)': 'price'}, axis='columns')
spot_price = spot_price.reset_index()
spot_price['index'] = pd.to_datetime(
    spot_price['index'].str[:-3], format='%Y %m%d')
spot_price['Date'] = pd.to_datetime(spot_price['index'])
spot_price.set_index('Date', inplace=True)
spot_price = spot_price.loc['2000-01-01':, ['price']]
spot_price = spot_price.astype(float)
print(spot_price)


print('Historical Spot price visualization:')
plt.figure(figsize=(15, 5))
plt.plot(spot_price)
plt.title('Henry Hub Spot Price (Daily frequency)')
plt.xlabel('Date_time')
plt.ylabel('Price ($/Mbtu)')
plt.show()


print('Missing values:', spot_price.isnull().sum())
# checking missing values
spot_price = spot_price.dropna()
# dropping missing valies
print('....Dropped Missing value row....')
print('Rechecking Missing values:', spot_price.isnull().sum())
# checking missing values


# Generate a Boxplot
print('Box plot visualization:')
spot_price.plot(kind='box', figsize=(10, 4))
plt.show()


# Generate a Histogram plot
print('Histogram visualization:')
spot_price.plot(kind='hist', figsize=(10, 4))
plt.show()


fig, ax1 = plt.subplots(ncols=1, figsize=(8, 5))
ax1.set_title('Price data before scaling')
sns.kdeplot(spot_price['price'], ax=ax1)
plt.show()
