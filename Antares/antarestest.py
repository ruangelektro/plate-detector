from antares_http import antares

antares.setDebug(True)
antares.setAccessKey('Privaet Key')

myData = {
    'temp' : 100,
    'windsp' : 10
}

antares.send(myData, 'Project name', "Application name")