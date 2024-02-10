CREATE SCHEMA ShirtShop
	CREATE TABLE Customers (
		CustomerID int IDENTITY(1,1) NOT NULL PRIMARY KEY, 
		FirstName varchar(50) NOT NULL, 
		LastName varchar(50) NOT NULL, 
		Email varchar(50) NOT NULL, 
		Phone varchar(50), 
		Address varchar(255)
	)
	CREATE TABLE Brands (
		BrandID int IDENTITY(1,1) NOT NULL PRIMARY KEY,
		BrandName varchar(50) NOT NULL,
		Address varchar(255),
		Phone varchar(50)
	)
	CREATE TABLE ShirtColors (
		ColorID int IDENTITY(1,1) Not NULL PRIMARY KEY,
		ColorName varchar(50) NOT NULL
	)
	CREATE TABLE ShirtSizes (
		SizeID int IDENTITY(1,1) NOT NULL PRIMARY KEY,
		Size varchar(50) NOT NULL
	)
	CREATE TABLE Products (
		ProductID int IDENTITY(1,1) NOT NULL PRIMARY KEY,
		BrandID int FOREIGN KEY REFERENCES Brands(BrandID),
		ProductName varchar(255) NOT NULL,
		Price float NOT NULL,
		Description varchar(255)
	)
	CREATE TABLE ProductVariants (
		VariantID int IDENTITY(1,1) NOT NULL PRIMARY KEY,
		ProductID int NOT NULL FOREIGN KEY REFERENCES Products(ProductID),
		ColorID int FOREIGN KEY REFERENCES ShirtColors(ColorID),
		SizeID int FOREIGN KEY REFERENCES ShirtSizes(SizeID),
		Quantity int NOT NULL
	)
	CREATE TABLE Statuses (
		StatusID int IDENTITY(1,1) NOT NULL PRIMARY KEY,
		ShortTile varchar(50) NOT NULL,
		Description varchar(255)
	)
	CREATE TABLE Orders (
		OrderID int IDENTITY(1,1) NOT NULL PRIMARY KEY,
		CustomerID int NOT NULL FOREIGN KEY REFERENCES Customers(CustomerID),
		StatusID int FOREIGN KEY REFERENCES Statuses(StatusID),
		OrderDate datetime NOT NULL,
		ShipDate datetime,
		ArrivalDate datetime
	)
	CREATE TABLE OrderLines (
		OrderID int NOT NULL FOREIGN KEY REFERENCES Orders(OrderID),
		VariantID int NOT NULL FOREIGN KEY REFERENCES ProductVariants(VariantID),
		Quantity int NOT NULL,
		PRIMARY KEY (OrderID, VariantID)
	);
