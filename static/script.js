$('#file-upload').submit(function(event) {
    event.preventDefault();

    const data = new FormData($(this)[0]);
    $.ajax({
        url: 'upload',
        data: data,
        type:'POST',
        contentType: false,
        processData: false
    });
});
