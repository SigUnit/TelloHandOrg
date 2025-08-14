from djitellopy import Tello


def connectionTest():
    tello = Tello()

    try:
        tello.connect()

        tello.takeoff()

        tello.streamon()

        tello.land()

        tello.streamoff()

        tello.end()
    except Exception as exception:
        print(exception)


if __name__ == '__main__':
    connectionTest()
