import os 
import json

# File to Store Library Data
LIBRARY_FILE = "library.txt"

# Loading Library Data from File
def load_library():
    if not os.path.exists(LIBRARY_FILE):
        return []
    try:
        with open(LIBRARY_FILE, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []
    
# Saving File Data to file
def save_library(library):
    with open(LIBRARY_FILE, "w") as file:
        return json.dump(library, file, indent=4)
    
# Adding a book from library
def add_book(library):
    title = input("Enter Title of your book! ")
    author = input("Enter Author of your book! ")
    year = input("Enter Year of your book! ")
    genre = input("Enter Genre of your book! ")
    read_status = input("Have you read this book before? ").strip().lower() == "yes"

    library.append({"title": title, "author" : author, "year" : year, "genre": genre, "read" : read_status})
    save_library(library)
    print("Book added successfully")

# Remove a book from library 
def remove_book(library):
    title = input("Enter the Title of the book you want to remove:")

    for book in library:
        if book["title"].lower() == title.lower():
            library.remove(book)
            save_library(library)
            print("Book has been removed successfully!\n")
            return
    
    print("Book not found! \n")

# Search for book in the library
def search_book(library):
    print("Search By:\n1. Title\n2. Author")
    choice = input("Enter your choice: ")

    if choice == "1":
        search_term = input("Enter the title").lower()
        matches = [book for book in library if search_term in book["title"].lower()]
    elif choice == "2":
       search_term = input("Enter the Author").lower()
       matches = [book for book in library if search_term in book["author"].lower()]
    else:
        print("Invalid Choice. \n")
        return
    
    if matches:
        print("Matching Books:")
        for i, book in enumerate(matches, 1):
            print(f"{i}. {book["title"]} by {book["author"]} ({book["year"]}) - {book["genre"]} - {"Read" if book["read"] else "Unread"}")

    else:
        print("No Matching books found.")
    print()

# Display All books in the library
def display_books(library):
    if not library:
        print("Your Library is Empty.")
        return
    
    print("Your Library:")
    for i, book in enumerate(library, 1):
        print(f"{i}. {book["title"]} by {book["author"]} ({book["year"]}) - {book["genre"]} - {"Read" if book["read"] else "Unread"}")
    print()

# Display Library statistics
def display_statistics(library):
    total_books = len(library)
    read_books = sum(1 for book in library if book["read"])
    read_percentage = (read_books / total_books * 100) if total_books > 0 else 0

    print(f"Total Books: {total_books}")
    print(f"Percentage read: {read_percentage:.1f}%\n")

# Main Menu 
def main():
    library = load_library()

    while True:
       print("Menu")
       print("1. Add a book")
       print("2. Remove a book")
       print("3. Search for a book")
       print("4. Display all books")
       print("5. Display Statistics")
       print("Exit.")

       choice = input("Enter your choice:")
       print()

       if choice == "1":
           add_book(library)
       elif choice == "2":
           remove_book(library)
       elif choice == "3":
           search_book(library)
       elif choice == "4":
           display_books(library)
       elif choice == "5":
           display_statistics(library)
       elif choice == "6":
           save_library(library)
           print("Library Saved to the file. Goodbye!")
           break
       else:
           print("Invalid choice. Please try again. \n")

if __name__ == "__main__":
    main()