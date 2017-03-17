$('#file-upload').submit(function(event) {
    event.preventDefault();

    const data = new FormData($(this)[0]);
    $.ajax({
        url: 'upload',
        data: data,
        type:'POST',
        contentType: false,
        processData: false,
        success: function(data) {
            // Display image and audio file
            const image = document.createElement('img');
            $(image).attr('src', data.image);

            const audio = document.createElement('audio');
            $(audio).attr('controls', 'controls');
            const audio_source = document.createElement('source');
            $(audio_source).attr('src', data.audio);
            $(audio_source).attr('type', 'audio/wav');
            $(audio).append(audio_source);

            $('#results').append(image, audio);
        }
    });
});
