
print("===== COLLEGE ADMISSION CHATBOT =====")

courses = ["BCA", "BBA", "BTech", "MBA"]
fees = {"bca": 70000, "bba": 65000, "btech": 100000, "mba": 90000}

while True:
    user = input("\nAsk about admission (or type bye): ").lower().strip()

    if user == "bye" or user == "exit":
        print("Thank you for visiting!")
        break

    elif "course" in user:
        print("Available courses:", courses)

    elif "fee" in user:
        found = False
        for course in fees:
            if course in user:
                print("Annual fee for", course.upper(), "is Rs.", fees[course])
                found = True
                break
        if not found:
            print("Please mention a course, e.g. BCA fee")

    elif "admission" in user or "process" in user:
        print("Admission steps:")
        print("1. Fill the application form")
        print("2. Submit required documents")
        print("3. Pay the admission fee")

    elif "document" in user:
        print("Documents: ID proof, marksheets, photos")

    elif "facility" in user:
        print("Facilities: Library, Wi-Fi, labs, sports")

    elif "contact" in user:
        print("Email: admissions@college.edu")
        print("Phone: 9876543210")

    else:
        print("Sorry, I don't understand. Try another query.")
