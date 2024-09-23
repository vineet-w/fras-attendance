from flask import Flask, send_file, request, render_template_string
import cv2
import face_recognition
import csv
import datetime
import os
import threading
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


app = Flask(__name__)

# Path to the folder containing known face images
known_faces_folder = "input_images"

# Load known face images and their corresponding names
known_face_images = []
known_face_names = []

for filename in os.listdir(known_faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        known_image_path = os.path.join(known_faces_folder, filename)
        known_face_images.append(face_recognition.load_image_file(known_image_path))
        known_face_names.append(os.path.splitext(filename)[0])  # Get the name without extension

# Initialize CSV file
csv_file = "recognized_faces.csv"
header = ["Name", "Timestamp"]

# Initialize an empty list to store known face encodings
known_face_encodings = []

for image in known_face_images:
    known_face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(known_face_encoding)

# Variables for frame skipping
frame_skip = 10
skip_counter = 0

# Flag to indicate if attendance is running
is_attendance_running = False

# Function to process frames
def process_frames():
    global skip_counter, is_attendance_running

    while is_attendance_running:
        # Read a single frame
        ret, frame = video_capture.read()

        # Skip frames if necessary
        skip_counter += 1
        if skip_counter < frame_skip:
            continue
        skip_counter = 0

        # Resize frame for better performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Find all face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Initialize name as "Unknown" for each detected face
            name = "Unknown"

            # Compare the detected face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

            if any(matches):
                # Get the name of the first known face that matches
                name = known_face_names[matches.index(True)]

                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Append the recognized face and timestamp to the CSV file
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, timestamp])

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/')
def index():
    return '''
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Main Page</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    </head>
    <body>
        <header>
            <h1 style="text-align:center">Welcome to Face Recognition Based Attendance System </h1>
            </header>
        <main style="text-align:center">
            
            <button class="btn btn-primary" id="start-attendance">Start Attendance</button>
            <button class="btn btn-primary" id="stop-attendance">Stop Attendance and Download</button>
        </main>
        <br>
        <section>
            <div class="header">
              <div style="background-image: url('https://preview.redd.it/0i6fg8op8fe81.jpg?width=1024&format=pjpg&auto=webp&s=321e2e05c6caab8e8f8cab07cccbae6c983ea9c8'); background-repeat: no-repeat; background-size: cover;">
            <div id="bigbox" class="position-relative overflow-hidden p-3 p-md-5 m-md-3 text-center bg-body-tertiary"  style="background-image: url('https://preview.redd.it/0i6fg8op8fe81.jpg?width=1024&format=pjpg&auto=webp&s=321e2e05c6caab8e8f8cab07cccbae6c983ea9c8'); background-repeat: no-repeat; background-size: cover;">
               
                <div class="col-md-6 p-lg-5 mx-auto my-5">
                  <h1 class="display-3 fw-bold text-white">F.R.A.S</h1>
                  <h3 class="fw-normal text-muted mb-3 text-white" style="color: #f8f9fa !important">face recognition based attendance system</h3>
                  <div class="d-flex gap-3 justify-content-center lead fw-normal">
                   
                   

                  </div>
                </div>
               
        </div>
        <section class="heros">
          <div class="container px-4 py-5" id="icon-grid">
            <h2 class="pb-2 border-bottom text-white">FEATURES</h2>
        
            <div class="row row-cols-1 row-cols-sm-2 row-cols-md-3 row-cols-lg-4 g-4 py-5">
              <div class="col d-flex align-items-start">
                <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Accuracy</h3>
                  <p class="text-white">Face recognition technology offers high accuracy in identifying individuals, reducing the chances of errors compared to manual attendance marking or other biometric systems.</p>
                </div>
              </div>
              <div class="col d-flex align-items-start">
                 <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Efficiency</h3>
                  <p class="text-white">It streamlines the attendance process by eliminating the need for physical cards, tokens, or manual entry, saving time for both teachers and students.</p>
                </div>
              </div>
              <div class="col d-flex align-items-start">
                <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Security</h3>
                  <p class="text-white">Face recognition provides a secure method of authentication, as each person's facial features are unique and difficult to replicate or falsify..</p>
                </div>
              </div>
              <div class="col d-flex align-items-start">
                <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Automation</h3>
                  <p class="text-white"> Automated attendance tracking reduces administrative overhead, allowing teachers to focus more on teaching tasks rather than attendance management.</p>
                </div>
              </div>
              <div class="col d-flex align-items-start">
                <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Contactless</h3>
                  <p class="text-white"> Unlike fingerprint or handprint-based systems, face recognition is contactless, reducing hygiene concerns, especially in situations like pandemics.</p>
                </div>
              </div>
              <div class="col d-flex align-items-start">
                 <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Integration</h3>
                  <p class="text-white"> They can be integrated with existing attendance management systems, making it easier to transition from traditional methods to biometric-based solutions..</p>
                </div>
              </div>
              <div class="col d-flex align-items-start">
                <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Non-intrusive</h3>
                  <p class="text-white">Students and teachers can easily enroll in the system without the need for physical contact or invasive procedures, enhancing user acceptance..</p>
                </div>
              </div>
              <div class="col d-flex align-items-start">
                 <div>
                  <h3 class="fw-bold mb-0 fs-4   text-white">Scalability</h3>
                  <p class="text-white">Face recognition systems can scale to accommodate large numbers of users without significant infrastructure changes, making them suitable for educational institutions of varying sizes.</p>
                </div>
              </div>
            </div>
          </div>
        </section> </div>
         <section>
          <div class="container">
            <hr>
            <footer class="py-5">
              <div class="row">
                <div class="col-6 col-md-2 mb-3">
                  <h5>Vineet Wagh</h5>
                  <ul class="nav flex-column">
                    <li class="nav-item mb-2">D6</a></li>
                    <li class="nav-item mb-2">68</a></li>
                    <li class="nav-item mb-2">Software Team</a></li>
                    
                  </ul>
                </div>
          
                <div class="col-6 col-md-2 mb-3">
                  <h5>Srushti Chopade</h5>
                  <ul class="nav flex-column">
                    <li class="nav-item mb-2">D6</a></li>
                    <li class="nav-item mb-2">15</a></li>
                    <li class="nav-item mb-2">Software Team</a></li>
                   
                  </ul>
                </div>
          
                <div class="col-6 col-md-2 mb-3">
                  <h5>Vighnesh Padwal</h5>
                  <ul class="nav flex-column">
                    <li class="nav-item mb-2">D6</a></li>
                    <li class="nav-item mb-2">41</a></li>
                    <li class="nav-item mb-2">Hardware Team</a></li>
                 
                  </ul>
                </div>

                <div class="col-6 col-md-2 mb-3">
                  <h5>Sneha Patil</h5>
                  <ul class="nav flex-column">
                    <li class="nav-item mb-2">D6</a></li>
                    <li class="nav-item mb-2">47</a></li>
                    <li class="nav-item mb-2">Hardware Team</a></li>
                   
                  </ul>
                </div>
               
                <br>
                <hr>
          
                <div class="">
                  <form>
                    <h5>CONTACT US </h5>
                   
                    <div class="d-flex flex-column flex-sm-row w-100 gap-2">
                      <label for="newsletter1" class="visually-hidden">Email address</label>
                      
                      <a class="btn btn-primary" href="mailto:vineetwagh45@gmail.com">Submit</a>
                     
                    </div>
                  </form>
                  <div style="text-align: center;">
                       <div style="display: inline-block;">
                        <img src="https://vesit.ves.ac.in/navlogo.PNG" alt="Image" style="width: 400px; height: 100px;">
                       </div>
                  </div>
                </div>
              </div>
          
              <div class="d-flex flex-column flex-sm-row justify-content-between py-4 my-4 border-top style="text-align: center;">
                
                <div style="display: inline-block;"><p style="align-centre">© 2024 F.R.A.S</p></div>
              </div>
            </footer>
          </div>
        </section>
        <script>
            let isAttendanceRunning = false;

            document.getElementById("start-attendance").addEventListener("click", function() {
                if (!isAttendanceRunning) {
                    fetch('/start-attendance')
                        .then(response => response.text())
                        .then(data => {
                            console.log(data);
                            isAttendanceRunning = true;
                        })
                        .catch(error => console.error('Error:', error));
                }
            });
            document.getElementById("stop-attendance").addEventListener("click", function() {
                if (isAttendanceRunning) {
                    let email = prompt("Please enter your email address to send the attendance:");
                    if (email) {
                        fetch(`/stop-attendance?email=${email}`)
                            .then(response => response.text())
                            .then(data => {
                                console.log(data);
                                isAttendanceRunning = false;
                                alert(data);
                            })
                            .catch(error => console.error('Error:', error));
                    }
                }
            });
        </script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
    </body>
    </html>
    '''

@app.route('/start-attendance')
def start_attendance():
    global is_attendance_running
    if not is_attendance_running:
        is_attendance_running = True
        threading.Thread(target=process_frames).start()
        return "Attendance started."
    return "Attendance is already running."

@app.route('/stop-attendance')
def stop_attendance():
    global is_attendance_running
    if is_attendance_running:
        is_attendance_running = False
        
        # Remove duplicates before sending or downloading the attendance
        remove_duplicates_from_csv()
        
        email = request.args.get('email')
        if email:
            send_attendance_email(email)
            return f"Attendance stopped and emailed to {email}."
        return "Attendance stopped."
    return "Attendance is not running."



# Function to remove duplicates from the CSV file
def remove_duplicates_from_csv():
    if os.path.exists(csv_file):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Drop duplicates based on 'Name' and 'Timestamp'
        df.drop_duplicates(subset=['Name', 'Timestamp'], inplace=True)
        
        # Rewrite the CSV file without duplicates
        df.to_csv(csv_file, index=False)
 
def send_attendance_email(email):
    # Email configurations
    sender_email = "2022.vineet.wagh@ves.ac.in"
    sender_password = "mktdhwouondwvhpc"
    subject = "Attendance Report"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject

    body = "Please find the attached attendance report."
    msg.attach(MIMEText(body, 'plain'))

    # Attach the CSV file
    attachment = open(csv_file, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename= {csv_file}")
    msg.attach(part)

    # Connect and send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    text = msg.as_string()
    server.sendmail(sender_email, email, text)
    server.quit()


@app.route('/download-attendance')
def download_attendance():
    return send_file(csv_file, as_attachment=True)

if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    app.run(debug=True, port=5500)