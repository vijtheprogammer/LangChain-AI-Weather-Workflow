import csv

def parse_zip_codes(file_path="Zip_Codes.csv"):
    zip_codes = set()  # use set to avoid duplicates

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            zip_code = row.get("ZIPCODE")
            if zip_code and zip_code.isdigit():  # ensure it's a numeric zipcode
                zip_codes.add(zip_code)

    # Return sorted list of zip codes as strings
    return sorted(zip_codes, key=int)

if __name__ == "__main__":
    zip_list = parse_zip_codes()
    print(zip_list)
