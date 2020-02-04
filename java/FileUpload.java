package cz.ivoa.vocloud.rest;

import cz.ivoa.vocloud.ejb.TokenAuthBean;
import cz.ivoa.vocloud.entity.AuthToken;

import javax.ejb.EJB;
import javax.ejb.Stateless;
import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import cz.ivoa.vocloud.filesystem.FilesystemManipulator;
import java.io.IOException;
import java.io.InputStream;
import java.io.FileInputStream;
import java.io.File;
import java.io.ByteArrayInputStream;

/**
 * Created by radiokoza on 1.4.17.
 */

@Stateless
@Path("/file")
public class FileUpload {
    @EJB
    private FilesystemManipulator fsm;

    @Path("/upload")
    @POST
    @Consumes("text/plain")
    @Produces("text/plain")
        public String uploadFile(String data,@QueryParam("folder") String folder,@QueryParam("filename") String filename, @QueryParam("force") Boolean force) throws IOException {
        InputStream dataStream = new ByteArrayInputStream(data.getBytes());
        fsm.saveUploadedFile(folder, filename, dataStream);
        return "ok";
    }
}