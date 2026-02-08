const int trigPin = 7;
const int echoPin = 6;

const int buzzerLeft = 2;
const int buzzerRight = 3;

const int motorPinA = 9;
const int motorPinB = 8;

String motorPosition = "center";

void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(buzzerLeft, OUTPUT);
  pinMode(buzzerRight, OUTPUT);
  pinMode(motorPinA, OUTPUT);
  pinMode(motorPinB, OUTPUT); 
  }

void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000);
  float distance = duration * 0.034 / 2;

  if (distance > 400 || distance == 0) distance = 0;

  Serial.println(distance);

  if (distance >= 1 && distance <= 10) {
    if (motorPosition == "left") {
      digitalWrite(buzzerLeft, HIGH);
      digitalWrite(buzzerRight, LOW);
    } 
    else if (motorPosition == "right") {
      digitalWrite(buzzerLeft, LOW);
      digitalWrite(buzzerRight, HIGH);
    } 
    else {
      digitalWrite(buzzerLeft, HIGH);
      digitalWrite(buzzerRight, HIGH);
    }
  } else {
    digitalWrite(buzzerLeft, LOW);
    digitalWrite(buzzerRight, LOW);
  }

  delay(50);
}

void moveLeft() {
  motorPosition = "left";
  digitalWrite(motorPinA, HIGH);
  digitalWrite(motorPinB, LOW);
}

void moveRight() {
  motorPosition = "right";
  digitalWrite(motorPinA, LOW);
  digitalWrite(motorPinB, HIGH);
}

void moveCenter() {
  if (motorPosition != "center") {
    motorPosition = "center";
    digitalWrite(motorPinA, LOW);
    digitalWrite(motorPinB, LOW);
  }
}