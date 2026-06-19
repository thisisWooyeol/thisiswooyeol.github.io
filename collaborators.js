class CollaboratorManager {
  constructor() {
    this.collaborators = {};
    this.loadCollaborators();
  }

  async loadCollaborators() {
    try {
      const response = await fetch("./collaborators.json");
      const data = await response.json();
      this.collaborators = data.collaborators;
      this.updateCollaboratorLinks();
      this.restoreHashScroll();
    } catch (error) {
      console.error("Failed to load collaborators:", error);
    }
  }

  getCollaborator(id) {
    return this.collaborators[id];
  }

  createCollaboratorLink(id) {
    const collaborator = this.getCollaborator(id);
    if (!collaborator) return id;

    return `<a href="${collaborator.url}">${collaborator.name}</a>`;
  }

  updateCollaboratorLinks() {
    // Update all elements with data-collaborator attribute
    document.querySelectorAll("[data-collaborator]").forEach((element) => {
      const collaboratorId = element.getAttribute("data-collaborator");
      element.innerHTML = this.createCollaboratorLink(collaboratorId);
    });
  }

  restoreHashScroll() {
    if (!window.location.hash) return;

    const target = document.getElementById(window.location.hash.slice(1));
    if (target) target.scrollIntoView();
  }
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const collaboratorManager = new CollaboratorManager();

  window.addEventListener("publications:rendered", () => {
    collaboratorManager.updateCollaboratorLinks();
    collaboratorManager.restoreHashScroll();
  });

  window.addEventListener("load", () => {
    if (!window.location.hash) return;

    const target = document.getElementById(window.location.hash.slice(1));
    if (target) target.scrollIntoView();
  });
});
